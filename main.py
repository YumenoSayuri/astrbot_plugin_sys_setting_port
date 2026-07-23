import asyncio
import copy
import time
from sys import maxsize

from astrbot.api.all import *
from astrbot.core.message.components import Image, Reply, At, Plain
from astrbot.core.agent.message import TextPart
from astrbot.core.utils.session_waiter import session_waiter, SessionController, SessionFilter
from astrbot.api.event.filter import on_llm_request
from astrbot.core.provider.entities import ProviderRequest

class UserSessionFilter(SessionFilter):
    def filter(self, event: AstrMessageEvent) -> str:
        return f"{event.unified_msg_origin}_{event.get_sender_id()}"

@register("astrbot_plugin_sys_setting_port", "Nova", "1.2.0", "系统设置移植 - 多模态转述控制与自定义等待")
class SysSettingPortPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 主动回复相关状态持久化
        import os
        import json
        # 使用 AstrBot 标准的插件数据持久化目录
        data_dir = os.path.join(os.getcwd(), "data", "plugin_data", "astrbot_plugin_sys_setting_port")
        os.makedirs(data_dir, exist_ok=True)
        self.data_file = os.path.join(data_dir, "proactive_data.json")
        self.last_chat_records = self._load_data()
        self.proactive_monitor_task = asyncio.create_task(self._proactive_monitor_loop())

    def _load_data(self):
        import os
        import json
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"读取主动回复数据失败: {e}")
        return {}

    def _save_data(self):
        import json
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(self.last_chat_records, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"保存主动回复数据失败: {e}")

    async def terminate(self):
        """插件卸载时清理后台任务"""
        if self.proactive_monitor_task:
            self.proactive_monitor_task.cancel()
        # 彻底解决重载报错：不再调用 super().terminate()，因为基类的 terminate 可能是同步的，
        # 强行 await 会导致 TypeError: object NoneType can't be used in 'await' expression。
        # 只要我们清理了自己的任务，插件就能安全退出。

    # ==================== 主动回复机制 ====================
    def _is_dnd_time(self, dnd_str: str) -> bool:
        """检查当前时间是否在免打扰时段内"""
        if not dnd_str or "-" not in dnd_str:
            return False
        try:
            start_hour, end_hour = map(int, dnd_str.split("-"))
            import datetime
            current_hour = datetime.datetime.now().hour
            if start_hour < end_hour:
                return start_hour <= current_hour < end_hour
            else: # 跨天，例如 23-7
                return current_hour >= start_hour or current_hour < end_hour
        except Exception:
            return False

    async def _judge_proactive(self, session_id: str, provider_id: str, prompt: str) -> bool:
        """调用轻量级模型判定是否应该主动回复"""
        prov = self.context.get_provider_by_id(provider_id)
        if not prov:
            logger.warning(f"未找到配置的智能判定模型: {provider_id}，默认返回否。")
            return False
            
        try:
            # 获取最近的聊天记录给判定模型看
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(session_id)
            history_str = ""
            if curr_cid:
                conversation = await self.context.conversation_manager.get_conversation(session_id, curr_cid)
                if conversation and conversation.history:
                    if isinstance(conversation.history, list):
                        for msg in conversation.history[-20:]:
                            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                                history_str += f"{msg.role}: {msg.content}\n"
                            elif isinstance(msg, dict):
                                history_str += f"{msg.get('role', 'unknown')}: {msg.get('content', '')}\n"
                            else:
                                history_str += str(msg) + "\n"
                    elif isinstance(conversation.history, str):
                        history_str = conversation.history[-4000:] if len(conversation.history) > 4000 else conversation.history
            
            full_prompt = f"{prompt}\n\n【最近聊天记录】\n{history_str}"
            
            resp = await prov.text_chat(prompt=full_prompt)
            if resp and resp.completion_text:
                text = resp.completion_text.strip()
                if "是" in text:
                    return True
            return False
        except Exception as e:
            logger.error(f"智能判定模型调用异常: {e}")
            return False

    async def _proactive_monitor_loop(self):
        """后台巡逻任务，定期检查是否有超时的私聊会话"""
        import random
        while True:
            try:
                await asyncio.sleep(60) # 每分钟检查一次
                
                if not self.config.get("enable_proactive_chat", False):
                    continue
                    
                mode = self.config.get("proactive_mode", "纯随机模式")
                min_interval = int(self.config.get("proactive_min_interval", 30))
                max_interval = int(self.config.get("proactive_max_interval", 120))
                dnd_time = self.config.get("proactive_dnd_time", "23-7")
                max_unanswered = int(self.config.get("proactive_max_unanswered", 3))
                raw_prompt = self.config.get("proactive_prompt", "")
                judge_provider = self.config.get("proactive_judge_provider", "")
                judge_prompt = self.config.get("proactive_judge_prompt", "")
                
                current_time = time.time()
                
                # 1. 免打扰检查：如果在免打扰时间内，直接半停机，不浪费任何资源
                if self._is_dnd_time(dnd_time):
                    continue
                
                for session_id, record in list(self.last_chat_records.items()):
                    last_time = record["time"]
                    unanswered_count = record.get("unanswered_count", 0)
                    
                    # 2. 检查是否达到未回复上限
                    if max_unanswered > 0 and unanswered_count >= max_unanswered:
                        continue
                        
                    should_trigger = False
                    
                    if mode == "纯随机模式":
                        # 初始化或获取下一次随机触发时间
                        next_random_time = record.get("next_random_time")
                        if not next_random_time:
                            random_minutes = random.randint(min_interval, max_interval)
                            next_random_time = last_time + random_minutes * 60
                            self.last_chat_records[session_id]["next_random_time"] = next_random_time
                            
                        if current_time >= next_random_time:
                            should_trigger = True
                            
                    elif mode == "智能判定模式":
                        last_judge_time = record.get("last_judge_time", last_time)
                        
                        # 保底机制：达到最大间隔，强制触发
                        if current_time - last_time >= max_interval * 60:
                            logger.info(f"会话 {session_id} 达到最大间隔 {max_interval} 分钟，强制触发主动回复！")
                            should_trigger = True
                        # 判定机制：每隔最小间隔，调用模型判定
                        elif current_time - last_judge_time >= min_interval * 60:
                            logger.info(f"会话 {session_id} 达到最小间隔 {min_interval} 分钟，调用模型进行智能判定...")
                            is_suitable = await self._judge_proactive(session_id, judge_provider, judge_prompt)
                            if is_suitable:
                                logger.info(f"模型判定：适合主动聊天！")
                                should_trigger = True
                            else:
                                logger.info(f"模型判定：不适合主动聊天，继续等待。")
                                self.last_chat_records[session_id]["last_judge_time"] = current_time
                    
                    # 3. 执行触发逻辑
                    if should_trigger:
                        logger.info(f"私聊会话 {session_id} 触发主动回复！(当前未回复次数: {unanswered_count})")
                        
                        # 动态渲染提示词
                        import datetime
                        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        rendered_prompt = raw_prompt.replace("{{current_time}}", current_time_str).replace("{{unanswered_count}}", str(unanswered_count))
                        
                        # 核心重构：直接调用 LLM，不伪造用户消息，避免污染上下文
                        try:
                            # 获取当前会话的 provider
                            curr_prov = self.context.get_using_provider(session_id)
                            if curr_prov:
                                # 获取历史记录
                                curr_cid = await self.context.conversation_manager.get_curr_conversation_id(session_id)
                                conversation = None
                                if curr_cid:
                                    conversation = await self.context.conversation_manager.get_conversation(session_id, curr_cid)
                                
                                # 构造请求上下文
                                contexts = []
                                if conversation and conversation.history:
                                    if isinstance(conversation.history, list):
                                        contexts = conversation.history[-20:]
                                    elif isinstance(conversation.history, str):
                                        rendered_prompt = f"【历史记录】\n{conversation.history[-4000:]}\n\n【当前指令】\n{rendered_prompt}"
                                
                                # 调用大模型
                                resp = await curr_prov.text_chat(
                                    prompt=rendered_prompt,
                                    contexts=contexts if contexts else None
                                )
                                
                                if resp and resp.completion_text:
                                    reply_text = resp.completion_text.strip()
                                    
                                    # 构造回复结果
                                    result = MessageEventResult().message(reply_text)
                                    
                                    # 触发 OnDecoratingResultEvent 钩子，让其他插件（如语音、表情包、分段）处理
                                    from astrbot.core.platform.astr_message_event import AstrMessageEvent
                                    from astrbot.core.platform.message_type import MessageType
                                    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember, Group
                                    from astrbot.core.star.star_handler import EventType, star_handlers_registry
                                    
                                    parts = session_id.split(":")
                                    if len(parts) >= 3:
                                        platform_name = parts[0]
                                        msg_type_str = parts[1]
                                        target_id = parts[2]
                                        
                                        platform_inst = None
                                        for p in self.context.platform_manager.platform_insts:
                                            if p.meta().id == platform_name or p.meta().name == platform_name:
                                                platform_inst = p
                                                break
                                                
                                        if platform_inst:
                                            message_obj = AstrBotMessage()
                                            if "Friend" in msg_type_str:
                                                message_obj.type = MessageType.FRIEND_MESSAGE
                                            elif "Group" in msg_type_str:
                                                message_obj.type = MessageType.GROUP_MESSAGE
                                                message_obj.group = Group(group_id=target_id)
                                            else:
                                                message_obj.type = MessageType.FRIEND_MESSAGE
                                                
                                            message_obj.session_id = target_id
                                            message_obj.message = result.chain
                                            message_obj.self_id = "bot"
                                            message_obj.sender = MessageMember(user_id=target_id)
                                            message_obj.message_str = ""
                                            message_obj.raw_message = None
                                            message_obj.message_id = ""
                                            
                                            dummy_event = AstrMessageEvent(
                                                message_str="",
                                                message_obj=message_obj,
                                                platform_meta=platform_inst.meta(),
                                                session_id=target_id
                                            )
                                            dummy_event.set_result(result)
                                            
                                            # 极其关键：打上 LLM 回复的烙印，否则 nova_omni 等插件会忽略它
                                            setattr(dummy_event, "__is_llm_reply", True)
                                            
                                            handlers = star_handlers_registry.get_handlers_by_event_type(EventType.OnDecoratingResultEvent)
                                            for handler in handlers:
                                                try:
                                                    await handler.handler(dummy_event)
                                                except Exception as e:
                                                    logger.error(f"执行装饰钩子失败: {handler.handler_full_name}, 错误: {e}")
                                                    
                                            res = dummy_event.get_result()
                                            if res is not None and res.chain is not None:
                                                result.chain = res.chain
                                                
                                    # 发送消息
                                    success = await self.context.send_message(session_id, result)
                                    
                                    if success:
                                        # 只有发送成功，才更新状态并重新开始计时
                                        self.last_chat_records[session_id]["time"] = current_time
                                        self.last_chat_records[session_id]["unanswered_count"] = unanswered_count + 1
                                        if "next_random_time" in self.last_chat_records[session_id]:
                                            del self.last_chat_records[session_id]["next_random_time"]
                                        if "last_judge_time" in self.last_chat_records[session_id]:
                                            del self.last_chat_records[session_id]["last_judge_time"]
                                        self._save_data()
                                        
                                        # 将 AI 的回复存入数据库历史记录
                                        if curr_cid:
                                            try:
                                                from astrbot.core.agent.message import AssistantMessageSegment, UserMessageSegment, TextPart
                                                
                                                # 构造极短的假 User 消息作为占位符，避免污染上下文
                                                user_msg = UserMessageSegment(content=[TextPart(text="(系统触发主动聊天)")])
                                                assistant_msg = AssistantMessageSegment(content=[TextPart(text=reply_text)])
                                                
                                                # 使用官方推荐的 add_message_pair 方法
                                                await self.context.conversation_manager.add_message_pair(
                                                    cid=curr_cid,
                                                    user_message=user_msg,
                                                    assistant_message=assistant_msg
                                                )
                                                logger.info(f"已将主动回复存入历史记录 (session_id: {session_id})")
                                            except Exception as e:
                                                logger.error(f"保存主动回复历史记录失败: {e}")
                                    else:
                                        logger.error(f"主动回复发送失败: 未找到匹配的平台适配器 (session_id: {session_id})")
                                        
                        except Exception as e:
                            logger.error(f"主动回复调用 LLM 或发送失败: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"主动回复巡逻任务发生异常: {e}")

    @event_message_type(EventMessageType.PRIVATE_MESSAGE, priority=maxsize)
    async def handle_record_private_chat(self, event: AstrMessageEvent):
        """记录私聊的最后活跃时间，用于主动回复"""
        if not self.config.get("enable_proactive_chat", False):
            return
            
        # 过滤掉机器人自己发的消息（如果有的话）
        sender_id = str(event.get_sender_id())
        if sender_id == str(event.get_self_id()):
            return
            
        # 白名单检查
        allowed_users = self.config.get("proactive_allowed_users", [])
        if not allowed_users or sender_id not in allowed_users:
            return
            
        session_id = event.unified_msg_origin
        if session_id not in self.last_chat_records:
            self.last_chat_records[session_id] = {}
            
        self.last_chat_records[session_id]["time"] = time.time()
        self.last_chat_records[session_id]["unanswered_count"] = 0 # 用户发消息了，重置未回复计数
        
        # 清除旧的随机时间和判定时间，让它们在巡逻任务中重新生成
        if "next_random_time" in self.last_chat_records[session_id]:
            del self.last_chat_records[session_id]["next_random_time"]
        if "last_judge_time" in self.last_chat_records[session_id]:
            del self.last_chat_records[session_id]["last_judge_time"]
            
        self._save_data()

    # ==================== 自定义等待机制 ====================
    @event_message_type(EventMessageType.ALL, priority=maxsize - 2)
    async def handle_custom_empty_mention(self, event: AstrMessageEvent):
        """接管只 @ 机器人的等待逻辑"""
        if not self.config.get("enable_custom_waiter", False):
            return

        try:
            messages = event.get_messages()
            wake_prefix = self.context.get_config(umo=event.unified_msg_origin).get("wake_prefix", [])
            
            # 过滤掉 Reply 组件和空白的 Plain 组件
            filtered_messages = []
            reply_components = []
            for m in messages:
                if isinstance(m, Reply):
                    reply_components.append(m)
                    continue
                if isinstance(m, Plain) and not m.text.strip():
                    continue
                filtered_messages.append(m)

            # 判断是否只有 @ 机器人，或者只有唤醒词
            is_empty_mention = False
            if len(filtered_messages) == 1:
                if isinstance(filtered_messages[0], At) and str(filtered_messages[0].qq) == str(event.get_self_id()):
                    is_empty_mention = True
                elif isinstance(filtered_messages[0], Plain) and filtered_messages[0].text.strip() in wake_prefix:
                    is_empty_mention = True

            if is_empty_mention:
                waiter_timeout = int(self.config.get("waiter_timeout", 60))
                waiter_need_reply = self.config.get("waiter_need_reply", True)
                waiter_reply_text = self.config.get("waiter_reply_text", "想要问什么呢？😄")
                wake_on_timeout = self.config.get("wake_on_timeout", False)

                if waiter_need_reply:
                    yield event.plain_result(waiter_reply_text)

                @session_waiter(waiter_timeout)
                async def custom_empty_mention_waiter(
                    controller: SessionController,
                    wait_event: AstrMessageEvent,
                ):
                    # 收到新消息，将之前的 Reply 和 @ 塞到新消息开头
                    insert_idx = 0
                    for rc in reply_components:
                        wait_event.message_obj.message.insert(insert_idx, rc)
                        insert_idx += 1
                        
                    wait_event.message_obj.message.insert(
                        insert_idx,
                        At(qq=event.get_self_id(), name=event.get_self_id()),
                    )
                    new_event = copy.copy(wait_event)
                    # 重新推入事件队列
                    self.context.get_event_queue().put_nowait(new_event)
                    wait_event.stop_event()
                    controller.stop()

                try:
                    await custom_empty_mention_waiter(event, session_filter=UserSessionFilter())
                except TimeoutError:
                    if wake_on_timeout:
                        logger.info(f"等待超时，模拟用户发送唤醒消息...")
                        fake_event = copy.copy(event)
                        fake_event.message_obj = copy.copy(event.message_obj)
                        
                        bot_id = event.get_self_id()
                        fake_msg_str = f"[At:{bot_id}]"
                        fake_event.message_str = fake_msg_str
                        
                        new_message = []
                        for rc in reply_components:
                            new_message.append(rc)
                        new_message.append(At(qq=bot_id, name=bot_id))
                        new_message.append(Plain(fake_msg_str))
                        
                        fake_event.message_obj.message = new_message
                        # 保持原消息的 message_id，完美引用
                        fake_event.message_obj.message_id = event.message_obj.message_id
                        import time
                        fake_event.message_obj.timestamp = int(time.time())
                        fake_event.clear_result()
                        self.context.get_event_queue().put_nowait(fake_event)
                except Exception as e:
                    logger.error(f"自定义等待发生错误: {e}")
                finally:
                    # 无论如何，停止当前这个只有 @ 的事件继续传播
                    event.stop_event()

        except Exception as e:
            logger.error(f"handle_custom_empty_mention error: {e}")

    @event_message_type(EventMessageType.ALL, priority=maxsize - 3)
    async def handle_strip_quote_image(self, event: AstrMessageEvent):
        """【移花接木】将引用消息中的图片提取到主消息体中。
        既能绕过底层强行调用当前 LLM 进行引用转述的硬编码逻辑，
        又能让其他插件（如图生图）和多模态 LLM 正常获取到图片。"""
        quote_images = []
        quote_sources = []
        for comp in event.message_obj.message:
            if isinstance(comp, Reply) and comp.chain:
                sender_name = (getattr(comp, "sender_nickname", None) or "未知用户").strip()
                sender_id = getattr(comp, "sender_id", None)
                source_label = sender_name
                if sender_id and str(sender_id) not in source_label:
                    source_label = f"{source_label}（{sender_id}）"

                new_chain = []
                for c in comp.chain:
                    if isinstance(c, Image):
                        quote_images.append(c)
                        quote_sources.append(source_label)
                    else:
                        new_chain.append(c)
                comp.chain = new_chain
        
        if quote_images:
            # 将提取出的图片追加到当前消息末尾，伪装成普通图片附件
            for img in quote_images:
                event.message_obj.message.append(img)
            event.set_extra(
                "sys_setting_port_quote_sources",
                quote_sources,
            )

    # ==================== 多模态转述机制 ====================
    async def _try_caption(self, provider_id: str, prompt: str, image_urls: list, max_retries: int, retry_keywords: list) -> str:
        prov = self.context.get_provider_by_id(provider_id)
        if not prov:
            logger.warning(f"未找到配置的图片转述模型: {provider_id}")
            return ""

        for attempt in range(max_retries):
            try:
                logger.info(f"正在使用 {provider_id} 转述图片 (尝试 {attempt + 1}/{max_retries})...")
                # 将提示词作为 system_prompt 传给多模态模型，prompt 留空或给个占位符
                resp = await prov.text_chat(system_prompt=prompt, prompt="[图片]", image_urls=image_urls)
                
                if not resp or not resp.completion_text:
                    logger.warning(f"{provider_id} 返回为空，准备重试...")
                    continue
                    
                text = resp.completion_text.strip()
                
                # 检查是否包含拒绝/道歉关键词
                is_failed = False
                for kw in retry_keywords:
                    if kw.lower() in text.lower():
                        is_failed = True
                        logger.warning(f"{provider_id} 返回内容包含失败关键词 '{kw}'，准备重试。返回内容: {text}")
                        break
                        
                if not is_failed:
                    return text
                    
            except Exception as e:
                logger.error(f"{provider_id} 转述发生异常: {e}，准备重试...")
                
            # 重试前稍微等待一下，避免被 API 频率限制
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                
        logger.error(f"{provider_id} 经过 {max_retries} 次尝试后仍然失败。")
        return ""

    @on_llm_request()
    async def on_llm_req(self, event: AstrMessageEvent, req: ProviderRequest):
        # === 多模态转述逻辑 ===
        caption_provider_id = self.config.get("caption_provider_id", "")
        fallback_provider_id = self.config.get("fallback_provider_id", "")
        target_models = self.config.get("target_models", [])
        caption_prompt = self.config.get("caption_prompt", "请详细描述这张图片的内容，以便纯文本模型能够理解。")
        max_retries = int(self.config.get("max_retries", 3))
        retry_keywords = self.config.get("retry_keywords", ["抱歉", "对不起", "无法", "sorry", "apologize", "error", "失败", "不能"])

        # 获取当前使用的模型名称
        curr_prov = self.context.get_using_provider(event.unified_msg_origin)
        model_name = req.model
        if not model_name and curr_prov:
            if hasattr(curr_prov, "get_model"):
                model_name = curr_prov.get_model()
            elif hasattr(curr_prov, "provider_meta") and curr_prov.provider_meta:
                model_name = curr_prov.provider_meta.model

        # 提取当前消息中的图片
        image_urls = list(req.image_urls) if req.image_urls else []
        
        # 主动出击：再次扫描 event.message_obj.message，确保没有遗漏任何图片
        # （因为底层构建 req.image_urls 的时机可能在我们追加引用图片之前）
        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                path = await comp.convert_to_file_path()
                if path not in image_urls:
                    image_urls.append(path)

        if not image_urls:
            return
            
        # 更新 req.image_urls，确保多模态模型能看到所有图片
        req.image_urls = image_urls

        # 图片已从 Reply 中移到主消息链，补充原引用发送者，避免模型误以为图片由当前用户发送。
        quote_sources = event.get_extra("sys_setting_port_quote_sources", [])
        if quote_sources:
            source_text = "；".join(dict.fromkeys(quote_sources))
            source_part = TextPart(
                text=f"\n[引用图片来源：{source_text}。图片属于被引用消息中的原发送者，不是当前发言者。]"
            )
            req.extra_user_content_parts.insert(0, source_part)

        # 如果当前模型不在目标列表中，说明它是多模态模型，直接返回，让它自己看图
        if not model_name:
            return
            
        # 模糊匹配：只要 target_models 中的某个字符串是当前模型名称的子串，就触发
        is_match = any(target.lower() in model_name.lower() for target in target_models)
        if not is_match:
            return

        # 既然是纯文本模型，无论转述是否成功，都必须清空图片，防止底层报错
        req.image_urls = []

        # 如果没有配置转述模型，也只能放弃转述
        if not caption_provider_id:
            return

        # 尝试主模型
        caption = await self._try_caption(caption_provider_id, caption_prompt, image_urls, max_retries, retry_keywords)
        
        # 如果主模型失败，尝试兜底模型
        if not caption and fallback_provider_id:
            logger.info(f"主模型 {caption_provider_id} 失败，切换至兜底模型 {fallback_provider_id}...")
            caption = await self._try_caption(fallback_provider_id, caption_prompt, image_urls, max_retries, retry_keywords)

        if caption:
            
            # 清理底层可能已经添加的图片路径提示文本，以及底层引用图片转述失败的残留文本
            new_parts = []
            for part in req.extra_user_content_parts:
                if isinstance(part, TextPart):
                    # 清理普通图片附件提示
                    if "[Image Attachment: path" in part.text:
                        continue
                    # 清理底层引用消息处理残留（无论成功还是失败）
                    if "<Quoted Message>" in part.text and "[Image Caption in quoted message]" in part.text:
                        # 尝试把底层错误的转述结果替换掉，或者直接保留我们自己的转述
                        pass # 这里不 continue，因为我们需要保留引用消息的文本部分，只在后面追加我们的转述
                new_parts.append(part)
            
            # 将转述结果作为普通文本插入到用户消息中，并保留被引用者身份。
            caption_source = ""
            if quote_sources:
                source_text = "；".join(dict.fromkeys(quote_sources))
                caption_source = f"（图片来自被引用消息的 {source_text}，不是当前发言者）"
            new_parts.append(TextPart(text=f"\n[图片转述内容]{caption_source}: {caption}"))
            req.extra_user_content_parts = new_parts
            
            logger.info("图片转述成功并已作为普通消息插入。")
        else:
            logger.error("所有转述模型均失败，放弃转述。")
import asyncio
import copy
import datetime
import time
from collections import OrderedDict, defaultdict
from sys import maxsize

from astrbot.api.all import *
from astrbot.core.message.components import Image, Reply, At, Plain
from astrbot.core.agent.message import Message, TextPart
from astrbot.core.utils.session_waiter import session_waiter, SessionController, SessionFilter
from astrbot.api.event.filter import on_llm_request
from astrbot.core.provider.entities import ProviderRequest

class UserSessionFilter(SessionFilter):
    def filter(self, event: AstrMessageEvent) -> str:
        return f"{event.unified_msg_origin}_{event.get_sender_id()}"

@register("astrbot_plugin_sys_setting_port", "Nova", "2.1.3", "系统设置移植 - 群聊视觉上下文、多模态转述与自定义等待")
class SysSettingPortPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        import os
        import json
        data_dir = os.path.join(os.getcwd(), "data", "plugin_data", "astrbot_plugin_sys_setting_port")
        os.makedirs(data_dir, exist_ok=True)
        self.data_file = os.path.join(data_dir, "proactive_data.json")
        self.last_chat_records = self._load_data()
        self.group_visual_history = defaultdict(list)
        self.group_visual_locks = defaultdict(asyncio.Lock)
        self.image_caption_cache = OrderedDict()
        self.message_image_cache = OrderedDict()
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
        if self.proactive_monitor_task:
            self.proactive_monitor_task.cancel()

    def _group_context_enabled(self, event: AstrMessageEvent) -> bool:
        return bool(self.config.get("enable_group_visual_context", False) and event.get_group_id())

    def _visual_group_enabled(self, event: AstrMessageEvent) -> bool:
        if not self._group_context_enabled(event):
            return False
        group_id = str(event.get_group_id())
        allowed_groups = {str(item).strip() for item in self.config.get("group_visual_allowed_groups", []) if str(item).strip()}
        return group_id in allowed_groups

    @staticmethod
    def _image_cache_key(image: Image) -> str:
        return str(getattr(image, "url", None) or getattr(image, "file", None) or id(image))

    def _purge_expired_caption_cache(self):
        ttl_seconds = max(60, int(self.config.get("group_visual_cache_ttl_hours", 6)) * 3600)
        expire_before = time.time() - ttl_seconds
        for cache in (self.image_caption_cache, self.message_image_cache):
            expired_keys = [key for key, value in cache.items() if value.get("cached_at", 0) < expire_before]
            for key in expired_keys:
                cache.pop(key, None)

    def _cache_put(self, cache: OrderedDict, key: str, value: dict, limit: int):
        if not key: return
        self._purge_expired_caption_cache()
        cache.pop(key, None)
        value["cached_at"] = time.time()
        cache[key] = value
        while len(cache) > max(1, limit):
            cache.popitem(last=False)

    async def _caption_group_image(self, image: Image) -> str:
        self._purge_expired_caption_cache()
        cache_key = self._image_cache_key(image)
        cached = self.image_caption_cache.get(cache_key)
        if cached:
            self.image_caption_cache.move_to_end(cache_key)
            return cached["caption"]

        provider_id = self.config.get("group_visual_provider_id", "")
        if not provider_id: return ""

        prompt = self.config.get("group_visual_prompt", "请用简洁准确的中文描述图片中的主体、动作、场景、文字和重要细节，供群聊上下文理解。只输出图片描述。")
        max_retries = max(1, int(self.config.get("max_retries", 3)))
        try:
            path = await image.convert_to_file_path()
            caption = await self._try_caption(provider_id, prompt, [path], max_retries)
            if caption:
                cache_limit = int(self.config.get("group_visual_cache_size", 300))
                self._cache_put(self.image_caption_cache, cache_key, {"caption": caption}, cache_limit)
            return caption
        except Exception as e:
            logger.error(f"群聊图片理解失败: {e}")
            return ""

    async def _build_group_history_message(self, event: AstrMessageEvent):
        parts = []
        captions = []
        images = []
        for comp in event.get_messages():
            if isinstance(comp, Plain):
                if comp.text: parts.append(comp.text)
            elif isinstance(comp, Image):
                if self._visual_group_enabled(event):
                    caption = await self._caption_group_image(comp)
                    parts.append(f"[图片：{caption}]" if caption else "[图片：理解失败]")
                    captions.append(caption)
                else:
                    parts.append("[图片]")
                images.append(comp)
            elif isinstance(comp, At):
                target = comp.name or comp.qq
                parts.append(f"[At:{target}]")
            elif isinstance(comp, Reply):
                sender = comp.sender_nickname or comp.sender_id or "未知用户"
                quoted_text = (comp.message_str or "").strip()
                if quoted_text: parts.append(f"[引用 {sender}：{quoted_text}]")
                else: parts.append(f"[引用 {sender} 的消息]")

        content = " ".join(part.strip() for part in parts if part and part.strip())
        if not content: return None, captions, images

        sender_name = event.get_sender_name() or str(event.get_sender_id())
        sender_id = str(event.get_sender_id())
        timestamp = getattr(event.message_obj, "timestamp", None)
        try:
            time_text = datetime.datetime.fromtimestamp(int(timestamp)).strftime("%H:%M:%S")
        except (TypeError, ValueError, OSError):
            time_text = datetime.datetime.now().strftime("%H:%M:%S")
        record = f"[{sender_name}（{sender_id}）/{time_text}]: {content}"
        return record, captions, images

    def _get_group_visual_history_text(self, group_id: str, current_message_id: str = "") -> str:
        history = self.group_visual_history.get(group_id, [])
        records = [item["record"] for item in history if not current_message_id or item.get("message_id") != current_message_id]
        return "\n---\n".join(records)

    def _inject_group_chat_context(self, req: ProviderRequest, history_text: str) -> str:
        context_text = (
            "<group_chat_context>\n"
            "以下是该群最近的真实聊天时间线。图片描述位于原图片消息的位置；"
            "请依据发送者、时间、At 与引用关系判断对话对象，不要默认所有消息都在对你说。\n"
            f"{history_text}\n"
            "</group_chat_context>"
        )

        if req.contexts is None:
            req.contexts = []
        for message in req.contexts:
            if isinstance(message, Message):
                content = message.content
            elif isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = ""
            if isinstance(content, str) and "<group_chat_context>" in content:
                return "existing_temporary_context"

        try:
            temporary_message = Message(role="user", content=context_text)
            object.__setattr__(temporary_message, "_no_save", True)
            req.contexts.append(temporary_message)
            return "temporary_context"
        except Exception as e:
            logger.warning(f"临时群聊上下文消息构建失败，回退到 system_prompt: {e}")
            system_prompt = req.system_prompt or ""
            if "<group_chat_context>" not in system_prompt:
                req.system_prompt = f"{system_prompt}\n\n{context_text}\n" if system_prompt else context_text
            return "system_prompt_fallback"

    @event_message_type(EventMessageType.GROUP_MESSAGE, priority=maxsize - 1)
    async def handle_group_visual_context(self, event: AstrMessageEvent):
        if not self._group_context_enabled(event): return
        if str(event.get_sender_id()) == str(event.get_self_id()): return

        group_id = str(event.get_group_id())
        async with self.group_visual_locks[group_id]:
            record, captions, images = await self._build_group_history_message(event)
            if not record: return

            history = self.group_visual_history[group_id]
            history.append({"message_id": str(event.message_obj.message_id or ""), "record": record})
            max_messages = max(1, int(self.config.get("group_visual_max_messages", 80)))
            if len(history) > max_messages: del history[:-max_messages]

            message_id = str(event.message_obj.message_id or "")
            if message_id and images:
                cache_limit = int(self.config.get("group_visual_cache_size", 300))
                self._cache_put(self.message_image_cache, message_id, {"captions": captions, "sender_name": event.get_sender_name() or str(event.get_sender_id()), "sender_id": str(event.get_sender_id())}, cache_limit)

    def _is_dnd_time(self, dnd_str: str) -> bool:
        if not dnd_str or "-" not in dnd_str: return False
        try:
            start_hour, end_hour = map(int, dnd_str.split("-"))
            current_hour = datetime.datetime.now().hour
            if start_hour < end_hour: return start_hour <= current_hour < end_hour
            else: return current_hour >= start_hour or current_hour < end_hour
        except Exception: return False

    async def _judge_proactive(self, session_id: str, provider_id: str, prompt: str) -> bool:
        prov = self.context.get_provider_by_id(provider_id)
        if not prov: return False
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(session_id)
            history_str = ""
            if curr_cid:
                conversation = await self.context.conversation_manager.get_conversation(session_id, curr_cid)
                if conversation and conversation.history:
                    if isinstance(conversation.history, list):
                        for msg in conversation.history[-20:]:
                            if hasattr(msg, 'role') and hasattr(msg, 'content'): history_str += f"{msg.role}: {msg.content}\n"
                            elif isinstance(msg, dict): history_str += f"{msg.get('role', 'unknown')}: {msg.get('content', '')}\n"
                    elif isinstance(conversation.history, str): history_str = conversation.history[-4000:]
            full_prompt = f"{prompt}\n\n【最近聊天记录】\n{history_str}"
            resp = await prov.text_chat(prompt=full_prompt)
            return resp and resp.completion_text and "是" in resp.completion_text.strip()
        except Exception: return False

    async def _proactive_monitor_loop(self):
        import random
        while True:
            try:
                await asyncio.sleep(60)
                if not self.config.get("enable_proactive_chat", False): continue
                mode = self.config.get("proactive_mode", "纯随机模式")
                min_interval = int(self.config.get("proactive_min_interval", 30))
                max_interval = int(self.config.get("proactive_max_interval", 120))
                dnd_time = self.config.get("proactive_dnd_time", "23-7")
                max_unanswered = int(self.config.get("proactive_max_unanswered", 3))
                raw_prompt = self.config.get("proactive_prompt", "")
                judge_provider = self.config.get("proactive_judge_provider", "")
                judge_prompt = self.config.get("proactive_judge_prompt", "")
                current_time = time.time()
                if self._is_dnd_time(dnd_time): continue
                for session_id, record in list(self.last_chat_records.items()):
                    last_time = record["time"]
                    unanswered_count = record.get("unanswered_count", 0)
                    if max_unanswered > 0 and unanswered_count >= max_unanswered: continue
                    should_trigger = False
                    if mode == "纯随机模式":
                        next_random_time = record.get("next_random_time")
                        if not next_random_time:
                            next_random_time = last_time + random.randint(min_interval, max_interval) * 60
                            self.last_chat_records[session_id]["next_random_time"] = next_random_time
                        if current_time >= next_random_time: should_trigger = True
                    elif mode == "智能判定模式":
                        last_judge_time = record.get("last_judge_time", last_time)
                        if current_time - last_time >= max_interval * 60: should_trigger = True
                        elif current_time - last_judge_time >= min_interval * 60:
                            if await self._judge_proactive(session_id, judge_provider, judge_prompt): should_trigger = True
                            else: self.last_chat_records[session_id]["last_judge_time"] = current_time
                    if should_trigger:
                        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        rendered_prompt = raw_prompt.replace("{{current_time}}", current_time_str).replace("{{unanswered_count}}", str(unanswered_count))
                        try:
                            curr_prov = self.context.get_using_provider(session_id)
                            if curr_prov:
                                curr_cid = await self.context.conversation_manager.get_curr_conversation_id(session_id)
                                conversation = await self.context.conversation_manager.get_conversation(session_id, curr_cid) if curr_cid else None
                                contexts = conversation.history[-20:] if conversation and isinstance(conversation.history, list) else []
                                resp = await curr_prov.text_chat(prompt=rendered_prompt, contexts=contexts if contexts else None)
                                if resp and resp.completion_text:
                                    reply_text = resp.completion_text.strip()
                                    result = MessageEventResult().message(reply_text)
                                    from astrbot.core.platform.astr_message_event import AstrMessageEvent
                                    from astrbot.core.platform.message_type import MessageType
                                    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember, Group
                                    from astrbot.core.star.star_handler import EventType, star_handlers_registry
                                    parts = session_id.split(":")
                                    if len(parts) >= 3:
                                        platform_name, msg_type_str, target_id = parts[0], parts[1], parts[2]
                                        platform_inst = next((p for p in self.context.platform_manager.platform_insts if p.meta().id == platform_name or p.meta().name == platform_name), None)
                                        if platform_inst:
                                            message_obj = AstrBotMessage()
                                            if "Friend" in msg_type_str: message_obj.type = MessageType.FRIEND_MESSAGE
                                            elif "Group" in msg_type_str:
                                                message_obj.type = MessageType.GROUP_MESSAGE
                                                message_obj.group = Group(group_id=target_id)
                                            message_obj.session_id, message_obj.message, message_obj.self_id, message_obj.sender = target_id, result.chain, "bot", MessageMember(user_id=target_id)
                                            dummy_event = AstrMessageEvent(message_str="", message_obj=message_obj, platform_meta=platform_inst.meta(), session_id=target_id)
                                            dummy_event.set_result(result)
                                            setattr(dummy_event, "__is_llm_reply", True)
                                            for handler in star_handlers_registry.get_handlers_by_event_type(EventType.OnDecoratingResultEvent):
                                                try: await handler.handler(dummy_event)
                                                except Exception: pass
                                            res = dummy_event.get_result()
                                            if res and res.chain: result.chain = res.chain
                                    if await self.context.send_message(session_id, result):
                                        self.last_chat_records[session_id].update({"time": current_time, "unanswered_count": unanswered_count + 1})
                                        self.last_chat_records[session_id].pop("next_random_time", None)
                                        self.last_chat_records[session_id].pop("last_judge_time", None)
                                        self._save_data()
                                        if curr_cid:
                                            from astrbot.core.agent.message import AssistantMessageSegment, UserMessageSegment, TextPart
                                            await self.context.conversation_manager.add_message_pair(cid=curr_cid, user_message=UserMessageSegment(content=[TextPart(text="(系统触发主动聊天)")]), assistant_message=AssistantMessageSegment(content=[TextPart(text=reply_text)]))
                        except Exception: pass
            except asyncio.CancelledError: break
            except Exception: pass

    @event_message_type(EventMessageType.PRIVATE_MESSAGE, priority=maxsize)
    async def handle_record_private_chat(self, event: AstrMessageEvent):
        if not self.config.get("enable_proactive_chat", False): return
        sender_id = str(event.get_sender_id())
        if sender_id == str(event.get_self_id()): return
        allowed_users = self.config.get("proactive_allowed_users", [])
        if not allowed_users or sender_id not in allowed_users: return
        session_id = event.unified_msg_origin
        if session_id not in self.last_chat_records: self.last_chat_records[session_id] = {}
        self.last_chat_records[session_id].update({"time": time.time(), "unanswered_count": 0})
        self.last_chat_records[session_id].pop("next_random_time", None)
        self.last_chat_records[session_id].pop("last_judge_time", None)
        self._save_data()

    @event_message_type(EventMessageType.ALL, priority=maxsize - 2)
    async def handle_custom_empty_mention(self, event: AstrMessageEvent):
        if not self.config.get("enable_custom_waiter", False): return
        try:
            messages = event.get_messages()
            wake_prefix = self.context.get_config(umo=event.unified_msg_origin).get("wake_prefix", [])
            filtered_messages, reply_components = [], []
            for m in messages:
                if isinstance(m, Reply): reply_components.append(m)
                elif isinstance(m, Plain) and not m.text.strip(): continue
                else: filtered_messages.append(m)
            is_empty = len(filtered_messages) == 1 and ((isinstance(filtered_messages[0], At) and str(filtered_messages[0].qq) == str(event.get_self_id())) or (isinstance(filtered_messages[0], Plain) and filtered_messages[0].text.strip() in wake_prefix))
            if is_empty:
                waiter_timeout = int(self.config.get("waiter_timeout", 60))
                if self.config.get("waiter_need_reply", True): yield event.plain_result(self.config.get("waiter_reply_text", "想要问什么呢？😄"))
                @session_waiter(waiter_timeout)
                async def custom_empty_mention_waiter(controller: SessionController, wait_event: AstrMessageEvent):
                    for i, rc in enumerate(reply_components): wait_event.message_obj.message.insert(i, rc)
                    wait_event.message_obj.message.insert(len(reply_components), At(qq=event.get_self_id(), name=event.get_self_id()))
                    self.context.get_event_queue().put_nowait(copy.copy(wait_event))
                    wait_event.stop_event(); controller.stop()
                try: await custom_empty_mention_waiter(event, session_filter=UserSessionFilter())
                except TimeoutError:
                    if self.config.get("wake_on_timeout", False):
                        bot_id = event.get_self_id(); fake_msg = f"[At:{bot_id}]"
                        fake_event = copy.copy(event); fake_event.message_obj = copy.copy(event.message_obj)
                        fake_event.message_str, fake_event.message_obj.message = fake_msg, reply_components + [At(qq=bot_id, name=bot_id), Plain(fake_msg)]
                        fake_event.message_obj.timestamp, fake_event.clear_result()
                        self.context.get_event_queue().put_nowait(fake_event)
                finally: event.stop_event()
        except Exception: pass

    @event_message_type(EventMessageType.ALL, priority=maxsize - 3)
    async def handle_strip_quote_image(self, event: AstrMessageEvent):
        quote_images, quote_sources, quote_captions = [], [], []
        for comp in event.message_obj.message:
            if isinstance(comp, Reply) and comp.chain:
                sender_name = (getattr(comp, "sender_nickname", None) or "未知用户").strip()
                sender_id = getattr(comp, "sender_id", None)
                source_label = f"{sender_name}（{sender_id}）" if sender_id and str(sender_id) not in sender_name else sender_name
                cached = self.message_image_cache.get(str(comp.id))
                if cached: quote_captions.extend(c for c in cached.get("captions", []) if c)
                new_chain = []
                for c in comp.chain:
                    if isinstance(c, Image): quote_images.append(c); quote_sources.append(source_label)
                    else: new_chain.append(c)
                comp.chain = new_chain
        if quote_images:
            for img in quote_images: event.message_obj.message.append(img)
            event.set_extra("sys_setting_port_quote_images", quote_images)
            event.set_extra("sys_setting_port_quote_sources", quote_sources)
            event.set_extra("sys_setting_port_quote_captions", quote_captions)

    @staticmethod
    def _match_target_model(req: ProviderRequest, provider, target_models: list) -> tuple[bool, str, list[str]]:
        candidates = []
        provider_config = getattr(provider, "provider_config", {}) or {}
        raw_candidates = [
            getattr(req, "model", None),
            provider.get_model() if provider and hasattr(provider, "get_model") else None,
            provider_config.get("model"),
            provider_config.get("id"),
        ]
        if provider and hasattr(provider, "meta"):
            try:
                meta = provider.meta()
                raw_candidates.extend([getattr(meta, "model", None), getattr(meta, "id", None)])
            except Exception:
                pass
        for value in raw_candidates:
            text = str(value or "").strip()
            if text and text.lower() not in {item.lower() for item in candidates}:
                candidates.append(text)

        keywords = [str(item).strip() for item in target_models or [] if str(item).strip()]
        for keyword in keywords:
            if any(keyword.lower() in candidate.lower() for candidate in candidates):
                return True, keyword, candidates
        return False, "", candidates

    @staticmethod
    def _provider_log_name(provider_id: str, provider) -> str:
        model = provider.get_model() if provider and hasattr(provider, "get_model") else ""
        return f"{provider_id} ({model})" if model else provider_id

    @staticmethod
    def _inject_caption_text(event: AstrMessageEvent, req: ProviderRequest, caption_text: str) -> str:
        if event.is_private_chat():
            if req.extra_user_content_parts is None:
                req.extra_user_content_parts = []
            req.extra_user_content_parts.append(TextPart(text=f"\n{caption_text}"))
            return "private_persistent_user_content"
        req.system_prompt = f"{req.system_prompt or ''}\n{caption_text}\n"
        return "group_temporary_system_prompt"

    async def _try_caption(self, provider_id: str, prompt: str, image_urls: list, max_retries: int) -> str:
        prov = self.context.get_provider_by_id(provider_id)
        if not prov: return ""
        structured_enabled = self.config.get("enable_caption_structured", True)
        judge_enabled, judge_provider_id, judge_prompt_tmpl = self.config.get("enable_caption_judge", False), self.config.get("caption_judge_provider_id", ""), self.config.get("caption_judge_prompt", "")
        if structured_enabled: prompt += "\n请务必将最终的图片描述内容包裹在 <caption_result> 标签中。如果无法描述，请输出 <error>原因</error>。"
        for attempt in range(max_retries):
            try:
                resp = await asyncio.wait_for(prov.text_chat(system_prompt=prompt, prompt="[图片]", image_urls=image_urls), timeout=45.0)
                if not resp or not resp.completion_text: continue
                raw_text = resp.completion_text.strip(); caption = raw_text
                if structured_enabled:
                    import re
                    if "<error>" in raw_text: continue
                    match = re.search(r"<caption_result>(.*?)</caption_result>", raw_text, re.DOTALL)
                    if match: caption = match.group(1).strip()
                if judge_enabled and judge_provider_id and judge_prompt_tmpl:
                    judge_prov = self.context.get_provider_by_id(judge_provider_id)
                    if judge_prov:
                        j_resp = await asyncio.wait_for(judge_prov.text_chat(prompt=judge_prompt_tmpl.replace("{{caption}}", caption)), timeout=20.0)
                        if j_resp and j_resp.completion_text and "否" in j_resp.completion_text.strip(): continue
                if caption: return caption
            except Exception: pass
            if attempt < max_retries - 1: await asyncio.sleep(1.5)
        return ""

    @on_llm_request(priority=-maxsize)
    async def inject_group_context_finally(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._group_context_enabled(event):
            return
        group_id = str(event.get_group_id())
        history_text = self._get_group_visual_history_text(group_id, str(event.message_obj.message_id or ""))
        if not history_text:
            logger.debug(f"群聊上下文无可注入历史: group={group_id}")
            return
        injection_mode = self._inject_group_chat_context(req, history_text)
        context_position = next(
            (
                index
                for index, message in enumerate(req.contexts or [])
                if "<group_chat_context>"
                in str(message.content if isinstance(message, Message) else message.get("content", ""))
            ),
            -1,
        )
        logger.info(
            f"群聊上下文已最终注入: group={group_id}, mode={injection_mode}, "
            f"position={context_position}, records={history_text.count(chr(10) + '---' + chr(10)) + 1}, "
            f"chars={len(history_text)}"
        )

    @on_llm_request()
    async def on_image_caption_req(self, event: AstrMessageEvent, req: ProviderRequest):
        caption_provider_id, fallback_provider_id, target_models = self.config.get("caption_provider_id", ""), self.config.get("fallback_provider_id", ""), self.config.get("target_models", [])
        caption_prompt, max_retries = self.config.get("caption_prompt", "请详细描述这张图片的内容，以便纯文本模型能够理解。"), int(self.config.get("max_retries", 3))
        curr_prov = self.context.get_using_provider(event.unified_msg_origin)
        is_target_text_model, matched_keyword, model_candidates = self._match_target_model(req, curr_prov, target_models)
        image_urls = list(req.image_urls) if req.image_urls else []
        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                path = await comp.convert_to_file_path()
                if path not in image_urls: image_urls.append(path)
        if not image_urls: return
        req.image_urls = image_urls
        quote_sources, quote_captions, quote_images = event.get_extra("sys_setting_port_quote_sources", []), event.get_extra("sys_setting_port_quote_captions", []), event.get_extra("sys_setting_port_quote_images", [])
        inspect_keywords = self.config.get("quote_image_inspect_keywords", ["仔细看", "看原图", "看细节", "重新看", "重看", "再看"])
        wants_original = bool(quote_images) and not is_target_text_model and any(k and k in (req.prompt or "") for k in inspect_keywords)
        if quote_images and not quote_captions and not wants_original and self._visual_group_enabled(event) and self.config.get("group_visual_provider_id"):
            for image in quote_images:
                caption = await self._caption_group_image(image)
                if caption: quote_captions.append(caption)
        
        if quote_sources and not event.is_private_chat():
            req.system_prompt = (req.system_prompt or "") + f"\n[系统附加信息 - 引用图片来源：{'；'.join(dict.fromkeys(quote_sources))}。图片属于被引用消息中的原发送者，不是当前发言者。]\n"

        quote_paths = []
        for image in quote_images:
            try: quote_paths.append(await image.convert_to_file_path())
            except Exception: pass
        if quote_images and not wants_original:
            req.image_urls = [p for p in req.image_urls if p not in quote_paths]
            image_urls = [p for p in image_urls if p not in quote_paths]
        if quote_captions and not wants_original:
            cached_caption_text = (
                f"[被引用图片的既有描述（来自 {'；'.join(dict.fromkeys(quote_sources)) or '原发送者'}）]: "
                f"{'；'.join(quote_captions)}"
            )
            cached_injection_mode = self._inject_caption_text(event, req, cached_caption_text)
            logger.info(
                f"【图片转述｜成功】provider=群聊图片描述缓存, mode={cached_injection_mode}, "
                f"chars={len('；'.join(quote_captions))}, images={len(quote_images)}"
            )
            if not image_urls:
                return
        if not is_target_text_model:
            logger.debug(
                f"【图片转述｜未触发】未匹配纯文本模型关键词；candidates={model_candidates}, "
                f"keywords={[str(item).strip() for item in target_models or []]}"
            )
            return
        req.image_urls = []
        if not caption_provider_id:
            logger.warning(
                f"【图片转述｜失败】已匹配关键词 {matched_keyword}，但未配置多模态转述模型；"
                f"candidates={model_candidates}"
            )
            return

        caption_provider = self.context.get_provider_by_id(caption_provider_id)
        logger.info(
            f"【图片转述｜触发】目标关键词={matched_keyword}, current_models={model_candidates}, "
            f"caption_provider={self._provider_log_name(caption_provider_id, caption_provider)}, images={len(image_urls)}"
        )
        caption = await self._try_caption(caption_provider_id, caption_prompt, image_urls, max_retries)
        used_provider_id = caption_provider_id
        used_provider = caption_provider
        if not caption and fallback_provider_id:
            fallback_provider = self.context.get_provider_by_id(fallback_provider_id)
            logger.warning(
                f"【图片转述｜主模型失败】准备调用兜底模型 "
                f"{self._provider_log_name(fallback_provider_id, fallback_provider)}"
            )
            caption = await self._try_caption(fallback_provider_id, caption_prompt, image_urls, max_retries)
            used_provider_id = fallback_provider_id
            used_provider = fallback_provider
        if not caption:
            logger.error(
                f"【图片转述｜失败】所有转述模型均未返回有效描述；target={matched_keyword}, images={len(image_urls)}"
            )
            return

        req.extra_user_content_parts = [
            part for part in (req.extra_user_content_parts or [])
            if not (isinstance(part, TextPart) and "[Image Attachment: path" in part.text)
        ]
        caption_source = f"（图片来自被引用消息的 {'；'.join(dict.fromkeys(quote_sources))}，不是当前发言者）" if quote_sources else ""
        caption_text = f"[图片转述内容]{caption_source}: {caption}"
        injection_mode = self._inject_caption_text(event, req, caption_text)
        logger.info(
            f"【图片转述｜成功】provider={self._provider_log_name(used_provider_id, used_provider)}, "
            f"mode={injection_mode}, chars={len(caption)}, images={len(image_urls)}"
        )

import asyncio
import copy
import datetime
import os
import time
from sys import maxsize

import aiosqlite

from astrbot.api.all import *
from astrbot.core.message.components import Image, Reply, At, Plain
from astrbot.core.agent.message import Message, TextPart
from astrbot.core.utils.session_waiter import session_waiter, SessionController, SessionFilter
from astrbot.api.event.filter import on_llm_request
from astrbot.core.provider.entities import ProviderRequest

class UserSessionFilter(SessionFilter):
    def filter(self, event: AstrMessageEvent) -> str:
        return f"{event.unified_msg_origin}_{event.get_sender_id()}"

@register("astrbot_plugin_sys_setting_port", "Nova", "2.2.0", "系统设置移植 - 会话请求超时、群聊视觉上下文、多模态转述与自定义等待")
class SysSettingPortPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        import json
        data_dir = os.path.join(os.getcwd(), "data", "plugin_data", "astrbot_plugin_sys_setting_port")
        os.makedirs(data_dir, exist_ok=True)
        self.data_file = os.path.join(data_dir, "proactive_data.json")
        self.caption_db_path = os.path.join(data_dir, "group_image_captions.db")
        self.caption_db_ready = False
        self.caption_db_lock = asyncio.Lock()
        self.last_chat_records = self._load_data()
        self.request_watchdogs = {}
        self.request_watchdog_sequence = 0
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
        for entry in list(self.request_watchdogs.values()):
            watchdog_task = entry.get("watchdog_task")
            if watchdog_task and not watchdog_task.done():
                watchdog_task.cancel()
        self.request_watchdogs.clear()

    def _finish_request_watchdog(self, session_id: str, token: int):
        entry = self.request_watchdogs.get(session_id)
        if not entry or entry.get("token") != token:
            return
        self.request_watchdogs.pop(session_id, None)
        watchdog_task = entry.get("watchdog_task")
        if not entry.get("timed_out") and watchdog_task and not watchdog_task.done():
            watchdog_task.cancel()

    async def _request_timeout_watchdog(
        self,
        event: AstrMessageEvent,
        session_id: str,
        token: int,
        pipeline_task: asyncio.Task,
        timeout_seconds: int,
    ):
        try:
            await asyncio.sleep(timeout_seconds)
            entry = self.request_watchdogs.get(session_id)
            if (
                not entry
                or entry.get("token") != token
                or entry.get("pipeline_task") is not pipeline_task
                or pipeline_task.done()
            ):
                return

            entry["timed_out"] = True
            elapsed = time.monotonic() - entry["started_at"]
            logger.error(
                f"【会话请求超时｜强制终止】session={session_id}, token={token}, "
                f"limit={timeout_seconds}s, elapsed={elapsed:.1f}s；正在取消当前请求并释放会话锁"
            )
            event.set_extra("sys_setting_port_request_timed_out", True)
            pipeline_task.cancel()

            if self.config.get("request_timeout_send_notice", True):
                notice = str(
                    self.config.get(
                        "request_timeout_notice_text",
                        "这次请求处理超时，已强制结束。后续消息现在可以继续处理。",
                    )
                ).strip()
                if notice:
                    try:
                        await asyncio.wait_for(
                            event.send(event.plain_result(notice)),
                            timeout=15,
                        )
                    except Exception as e:
                        logger.warning(f"会话请求超时提示发送失败: {e}")
        except asyncio.CancelledError:
            return

    @on_llm_request(priority=maxsize)
    async def register_request_timeout_watchdog(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ):
        if not self.config.get("enable_request_timeout_watchdog", True):
            return
        timeout_seconds = max(
            1,
            int(self.config.get("request_timeout_seconds", 180)),
        )
        pipeline_task = asyncio.current_task()
        if pipeline_task is None:
            logger.warning("无法获取当前 pipeline task，会话请求超时看门狗未启动")
            return

        session_id = event.unified_msg_origin
        old_entry = self.request_watchdogs.pop(session_id, None)
        if old_entry:
            old_watchdog = old_entry.get("watchdog_task")
            if old_watchdog and not old_watchdog.done():
                old_watchdog.cancel()

        self.request_watchdog_sequence += 1
        token = self.request_watchdog_sequence
        entry = {
            "token": token,
            "pipeline_task": pipeline_task,
            "watchdog_task": None,
            "started_at": time.monotonic(),
            "timed_out": False,
        }
        self.request_watchdogs[session_id] = entry
        watchdog_task = asyncio.create_task(
            self._request_timeout_watchdog(
                event,
                session_id,
                token,
                pipeline_task,
                timeout_seconds,
            )
        )
        entry["watchdog_task"] = watchdog_task
        pipeline_task.add_done_callback(
            lambda _task, sid=session_id, tok=token: self._finish_request_watchdog(sid, tok)
        )
        logger.info(
            f"【会话请求看门狗｜已启动】session={session_id}, token={token}, limit={timeout_seconds}s"
        )

    def _group_context_enabled(self, event: AstrMessageEvent) -> bool:
        return bool(self.config.get("enable_group_visual_context", False) and event.get_group_id())

    def _visual_group_enabled(self, event: AstrMessageEvent) -> bool:
        if not self._group_context_enabled(event):
            return False
        group_id = str(event.get_group_id())
        allowed_groups = {
            str(item).strip()
            for item in self.config.get("group_visual_allowed_groups", [])
            if str(item).strip()
        }
        return group_id in allowed_groups

    @staticmethod
    def _group_cache_session_id(event: AstrMessageEvent) -> str:
        return str(event.get_group_id() or event.unified_msg_origin)

    async def _ensure_caption_db(self):
        if self.caption_db_ready:
            return
        async with self.caption_db_lock:
            if self.caption_db_ready:
                return
            async with aiosqlite.connect(self.caption_db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA busy_timeout=10000")
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS group_image_captions (
                        session_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        image_key TEXT NOT NULL,
                        caption TEXT NOT NULL,
                        sender_name TEXT NOT NULL DEFAULT '',
                        sender_id TEXT NOT NULL DEFAULT '',
                        created_at REAL NOT NULL,
                        PRIMARY KEY (session_id, message_id, image_key)
                    )
                    """
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_group_image_session_time "
                    "ON group_image_captions(session_id, created_at DESC)"
                )
                await db.commit()
            self.caption_db_ready = True

    def _caption_cache_ttl_seconds(self) -> int:
        hours = max(0, int(self.config.get("group_visual_cache_ttl_hours", 8)))
        return hours * 3600

    async def _get_captions_by_message_ids(
        self,
        session_id: str,
        message_ids: list[str],
    ) -> dict[str, list[str]]:
        unique_ids = list(dict.fromkeys(str(item) for item in message_ids if str(item)))
        if not session_id or not unique_ids:
            return {}
        await self._ensure_caption_db()
        ttl_seconds = self._caption_cache_ttl_seconds()
        placeholders = ",".join("?" for _ in unique_ids)
        query = (
            "SELECT message_id, caption FROM group_image_captions "
            f"WHERE session_id = ? AND message_id IN ({placeholders})"
        )
        params = [session_id, *unique_ids]
        if ttl_seconds > 0:
            query += " AND created_at >= ?"
            params.append(time.time() - ttl_seconds)
        query += " ORDER BY created_at ASC, rowid ASC"
        result = {}
        async with aiosqlite.connect(self.caption_db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
        for message_id, caption in rows:
            if caption:
                result.setdefault(str(message_id), []).append(str(caption))
        return result

    async def _get_message_captions(self, session_id: str, message_id: str) -> list[str]:
        result = await self._get_captions_by_message_ids(session_id, [message_id])
        return result.get(str(message_id), [])

    async def _save_message_captions(
        self,
        session_id: str,
        message_id: str,
        captions: list[tuple[str, str]],
        sender_name: str,
        sender_id: str,
    ):
        valid = [(str(key), str(caption).strip()) for key, caption in captions if str(caption).strip()]
        if not session_id or not message_id or not valid:
            return
        await self._ensure_caption_db()
        now = time.time()
        per_session_limit = max(1, int(self.config.get("group_visual_cache_size", 100)))
        ttl_seconds = self._caption_cache_ttl_seconds()
        async with self.caption_db_lock:
            async with aiosqlite.connect(self.caption_db_path) as db:
                await db.execute("PRAGMA busy_timeout=10000")
                await db.executemany(
                    """
                    INSERT INTO group_image_captions
                    (session_id, message_id, image_key, caption, sender_name, sender_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, message_id, image_key) DO UPDATE SET
                        caption=excluded.caption,
                        sender_name=excluded.sender_name,
                        sender_id=excluded.sender_id,
                        created_at=excluded.created_at
                    """,
                    [
                        (session_id, message_id, key, caption, sender_name, sender_id, now)
                        for key, caption in valid
                    ],
                )
                if ttl_seconds > 0:
                    await db.execute(
                        "DELETE FROM group_image_captions WHERE session_id = ? AND created_at < ?",
                        (session_id, now - ttl_seconds),
                    )
                await db.execute(
                    """
                    DELETE FROM group_image_captions
                    WHERE session_id = ? AND message_id NOT IN (
                        SELECT message_id FROM group_image_captions
                        WHERE session_id = ?
                        GROUP BY message_id
                        ORDER BY MAX(created_at) DESC
                        LIMIT ?
                    )
                    """,
                    (session_id, session_id, per_session_limit),
                )
                await db.commit()

    @staticmethod
    def _image_cache_key(image: Image, index: int = 0) -> str:
        return str(
            getattr(image, "url", None)
            or getattr(image, "file", None)
            or f"image_{index}"
        )

    async def _caption_group_image(self, image: Image) -> str:
        provider_id = self.config.get("group_visual_provider_id", "")
        if not provider_id:
            return ""
        prompt = self.config.get(
            "group_visual_prompt",
            "请用简洁准确的中文描述图片中的主体、动作、场景、文字和重要细节，供群聊上下文理解。只输出图片描述。",
        )
        max_retries = max(1, int(self.config.get("max_retries", 3)))
        try:
            path = await image.convert_to_file_path()
            return await self._try_caption(provider_id, prompt, [path], max_retries)
        except Exception as e:
            logger.error(f"群聊图片理解失败: {e}")
            return ""

    @staticmethod
    def _format_onebot_history_message(msg: dict, captions: list[str]) -> str:
        sender = msg.get("sender", {}) or {}
        sender_name = sender.get("card") or sender.get("nickname") or "未知用户"
        sender_id = str(sender.get("user_id", ""))
        try:
            time_text = datetime.datetime.fromtimestamp(int(msg.get("time", 0))).strftime("%H:%M:%S")
        except (TypeError, ValueError, OSError):
            time_text = "未知时间"
        parts = []
        caption_index = 0
        for segment in msg.get("message", []) or []:
            segment_type = str(segment.get("type", ""))
            data = segment.get("data", {}) or {}
            if segment_type == "text":
                text = str(data.get("text", "")).strip()
                if text:
                    parts.append(text)
            elif segment_type == "image":
                if caption_index < len(captions):
                    parts.append(f"[图片：{captions[caption_index]}]")
                else:
                    parts.append("[图片]")
                caption_index += 1
            elif segment_type == "at":
                parts.append(f"[At:{data.get('name') or data.get('qq') or '未知'}]")
            elif segment_type == "reply":
                parts.append(f"[引用消息:{data.get('id') or '未知'}]")
            elif segment_type == "face":
                parts.append(f"[表情:{data.get('id') or ''}]")
            elif segment_type == "record":
                parts.append("[语音]")
            elif segment_type == "video":
                parts.append("[视频]")
            elif segment_type == "file":
                parts.append(f"[文件:{data.get('name') or data.get('file') or '未知'}]")
        content = " ".join(part for part in parts if part).strip()
        if not content:
            return ""
        return f"[{sender_name}（{sender_id}）/{time_text}]: {content}"

    async def _load_latest_group_history(self, event: AstrMessageEvent) -> str:
        if event.get_platform_name() != "aiocqhttp":
            logger.warning("群聊即时上下文仅支持 aiocqhttp/NapCat，当前平台已跳过")
            return ""
        bot = getattr(event, "bot", None)
        api = getattr(bot, "api", bot)
        if not api or not hasattr(api, "call_action"):
            logger.warning("无法取得 NapCat API，群聊即时上下文已跳过")
            return ""
        group_id = str(event.get_group_id() or "")
        count = max(1, int(self.config.get("group_visual_max_messages", 80)))
        result = await asyncio.wait_for(
            api.call_action(
                "get_group_msg_history",
                group_id=int(group_id) if group_id.isdigit() else group_id,
                message_seq=0,
                count=count,
                reverseOrder=False,
            ),
            timeout=15,
        )
        raw_messages = result.get("messages", []) if isinstance(result, dict) else []
        current_message_id = str(event.message_obj.message_id or "")
        messages = [
            msg for msg in raw_messages
            if str(msg.get("message_id", "")) != current_message_id
        ]
        messages.sort(
            key=lambda msg: (
                int(msg.get("time", 0) or 0),
                str(msg.get("message_id", "")),
            )
        )
        message_ids = [str(msg.get("message_id", "")) for msg in messages]
        caption_map = await self._get_captions_by_message_ids(
            self._group_cache_session_id(event),
            message_ids,
        )
        records = []
        for msg in messages[-count:]:
            message_id = str(msg.get("message_id", ""))
            record = self._format_onebot_history_message(msg, caption_map.get(message_id, []))
            if record:
                records.append(record)
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
        if not self._visual_group_enabled(event):
            return
        if str(event.get_sender_id()) == str(event.get_self_id()):
            return
        images = [comp for comp in event.get_messages() if isinstance(comp, Image)]
        message_id = str(event.message_obj.message_id or "")
        if not images or not message_id:
            return
        cache_session_id = self._group_cache_session_id(event)
        existing = await self._get_message_captions(cache_session_id, message_id)
        if len(existing) >= len(images):
            return
        captions = []
        for index, image in enumerate(images):
            if index < len(existing):
                continue
            caption = await self._caption_group_image(image)
            if caption:
                captions.append((self._image_cache_key(image, index), caption))
        if captions:
            await self._save_message_captions(
                cache_session_id,
                message_id,
                captions,
                event.get_sender_name() or str(event.get_sender_id()),
                str(event.get_sender_id()),
            )
            logger.info(
                f"群聊静默图片描述已持久化: group={cache_session_id}, "
                f"message={message_id}, images={len(captions)}"
            )

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
                if event.get_group_id():
                    quote_captions.extend(
                        await self._get_message_captions(
                            self._group_cache_session_id(event),
                            str(comp.id),
                        )
                    )
                new_chain = []
                for component in comp.chain:
                    if isinstance(component, Image):
                        quote_images.append(component)
                        quote_sources.append(source_label)
                    else:
                        new_chain.append(component)
                comp.chain = new_chain
        if quote_images:
            for image in quote_images:
                event.message_obj.message.append(image)
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
        pending = list(event.get_extra("sys_setting_port_pending_captions", []))
        if caption_text not in pending:
            pending.append(caption_text)
        event.set_extra("sys_setting_port_pending_captions", pending)
        return "group_pending_temporary_context"

    @staticmethod
    def _inject_temporary_user_context(req: ProviderRequest, text: str) -> int:
        if req.contexts is None:
            req.contexts = []
        temporary_message = Message(role="user", content=text)
        object.__setattr__(temporary_message, "_no_save", True)
        req.contexts.append(temporary_message)
        return len(req.contexts) - 1

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
        if event.is_private_chat():
            return

        group_id = str(event.get_group_id() or "")
        if self._group_context_enabled(event):
            try:
                history_text = await self._load_latest_group_history(event)
            except Exception as e:
                history_text = ""
                logger.error(f"从 NapCat 获取最新群聊上下文失败: group={group_id}, error={e}")
            if history_text:
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
            else:
                logger.debug(f"群聊上下文无可注入历史: group={group_id}")

        pending_captions = event.get_extra("sys_setting_port_pending_captions", [])
        for caption_text in pending_captions:
            position = self._inject_temporary_user_context(
                req,
                f"<current_image_caption>\n{caption_text}\n</current_image_caption>",
            )
            logger.info(
                f"【图片转述｜最终注入】group={group_id}, mode=temporary_context, "
                f"position={position}, chars={len(caption_text)}"
            )

    @on_llm_request()
    async def on_image_caption_req(self, event: AstrMessageEvent, req: ProviderRequest):
        caption_provider_id = self.config.get("caption_provider_id", "")
        fallback_provider_id = self.config.get("fallback_provider_id", "")
        target_models = self.config.get("target_models", [])
        caption_prompt = self.config.get("caption_prompt", "请详细描述这张图片的内容，以便纯文本模型能够理解。")
        max_retries = int(self.config.get("max_retries", 3))
        curr_prov = self.context.get_using_provider(event.unified_msg_origin)
        is_target_text_model, matched_keyword, model_candidates = self._match_target_model(req, curr_prov, target_models)

        all_paths = list(req.image_urls or [])
        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                path = await comp.convert_to_file_path()
                if path not in all_paths:
                    all_paths.append(path)
        if not all_paths:
            return
        req.image_urls = all_paths

        quote_sources = event.get_extra("sys_setting_port_quote_sources", [])
        quote_captions = list(event.get_extra("sys_setting_port_quote_captions", []))
        quote_images = event.get_extra("sys_setting_port_quote_images", [])
        quote_paths = []
        for image in quote_images:
            try:
                path = await image.convert_to_file_path()
                if path not in quote_paths:
                    quote_paths.append(path)
            except Exception:
                pass
        current_paths = [path for path in all_paths if path not in quote_paths]
        source_text = "；".join(dict.fromkeys(quote_sources)) or "原发送者"
        inspect_keywords = self.config.get("quote_image_inspect_keywords", ["仔细看", "看原图", "看细节", "重新看", "重看", "再看"])
        wants_original = bool(quote_paths) and not is_target_text_model and any(
            keyword and keyword in (req.prompt or "") for keyword in inspect_keywords
        )

        if not is_target_text_model:
            if quote_paths and not wants_original:
                if not quote_captions and self._visual_group_enabled(event) and self.config.get("group_visual_provider_id"):
                    for image in quote_images:
                        caption = await self._caption_group_image(image)
                        if caption:
                            quote_captions.append(caption)
                req.image_urls = current_paths
                if quote_captions:
                    caption_text = f"[被引用图片的既有描述（来自 {source_text}）]: {'；'.join(quote_captions)}"
                    mode = self._inject_caption_text(event, req, caption_text)
                    logger.info(
                        f"【图片转述｜成功】provider=群聊图片描述缓存, mode={mode}, "
                        f"chars={len('；'.join(quote_captions))}, images={len(quote_paths)}"
                    )
            return

        req.image_urls = []
        req.extra_user_content_parts = [
            part for part in (req.extra_user_content_parts or [])
            if not (isinstance(part, TextPart) and "[Image Attachment: path" in part.text)
        ]
        if not caption_provider_id:
            logger.warning(
                f"【图片转述｜失败】已匹配关键词 {matched_keyword}，但未配置多模态转述模型；"
                f"candidates={model_candidates}"
            )
            return

        async def caption_batch(paths: list[str], kind: str, source: str = ""):
            if not paths:
                return
            caption_provider = self.context.get_provider_by_id(caption_provider_id)
            logger.info(
                f"【图片转述｜触发】kind={kind}, 目标关键词={matched_keyword}, "
                f"current_models={model_candidates}, caption_provider="
                f"{self._provider_log_name(caption_provider_id, caption_provider)}, images={len(paths)}"
            )
            caption = await self._try_caption(caption_provider_id, caption_prompt, paths, max_retries)
            used_provider_id, used_provider = caption_provider_id, caption_provider
            if not caption and fallback_provider_id:
                fallback_provider = self.context.get_provider_by_id(fallback_provider_id)
                logger.warning(
                    f"【图片转述｜主模型失败】kind={kind}, 准备调用兜底模型 "
                    f"{self._provider_log_name(fallback_provider_id, fallback_provider)}"
                )
                caption = await self._try_caption(fallback_provider_id, caption_prompt, paths, max_retries)
                used_provider_id, used_provider = fallback_provider_id, fallback_provider
            if not caption:
                logger.error(
                    f"【图片转述｜失败】kind={kind}, 所有转述模型均未返回有效描述；"
                    f"target={matched_keyword}, images={len(paths)}"
                )
                return
            source_note = f"（来自被引用消息的 {source}，不是当前发言者）" if source else "（当前消息图片）"
            caption_text = f"[图片转述内容]{source_note}: {caption}"
            mode = self._inject_caption_text(event, req, caption_text)
            logger.info(
                f"【图片转述｜成功】kind={kind}, provider="
                f"{self._provider_log_name(used_provider_id, used_provider)}, "
                f"mode={mode}, chars={len(caption)}, images={len(paths)}"
            )

        if quote_captions:
            cached_text = f"[被引用图片的既有描述（来自 {source_text}）]: {'；'.join(quote_captions)}"
            mode = self._inject_caption_text(event, req, cached_text)
            logger.info(
                f"【图片转述｜成功】kind=quoted_cached, provider=群聊图片描述缓存, "
                f"mode={mode}, chars={len('；'.join(quote_captions))}, images={len(quote_paths)}"
            )
        else:
            await caption_batch(quote_paths, "quoted_uncached", source_text)
        await caption_batch(current_paths, "current", "")

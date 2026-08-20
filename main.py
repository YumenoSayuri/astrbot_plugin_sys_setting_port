import asyncio
import base64
import copy
import datetime
import hashlib
import io
import os
import time
from collections import Counter
from sys import maxsize

import aiosqlite

from astrbot.api.all import *
from astrbot.core.message.components import Image, Reply, At, Plain
from astrbot.core.agent.message import Message, TextPart
from astrbot.core.utils.session_waiter import session_waiter, SessionController, SessionFilter
from astrbot.api.event.filter import (
    PermissionType,
    command,
    llm_tool,
    on_astrbot_loaded,
    on_llm_request,
    on_llm_response,
    on_using_llm_tool,
    on_llm_tool_respond,
    permission_type,
)
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

class UserSessionFilter(SessionFilter):
    def filter(self, event: AstrMessageEvent) -> str:
        return f"{event.unified_msg_origin}_{event.get_sender_id()}"

@register("astrbot_plugin_sys_setting_port", "Nova", "2.3.4", "系统设置移植 - 会话请求超时、群聊上下文、群角色感知、多模态转述与自定义等待")
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
        self.group_role_sync_lock = asyncio.Lock()
        self.last_chat_records = self._load_data()
        self.request_watchdogs = {}
        self.request_watchdog_sequence = 0
        self.proactive_monitor_task = asyncio.create_task(self._proactive_monitor_loop())
        self.group_role_sync_task = None
        self._ensure_group_role_sync_task()

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
        if self.group_role_sync_task:
            self.group_role_sync_task.cancel()
        for entry in list(self.request_watchdogs.values()):
            watchdog_task = entry.get("watchdog_task")
            if watchdog_task and not watchdog_task.done():
                watchdog_task.cancel()
        self.request_watchdogs.clear()

    def _finish_request_watchdog(
        self,
        session_id: str,
        token: int,
        finish_reason: str = "pipeline_done",
    ) -> bool:
        entry = self.request_watchdogs.get(session_id)
        if not entry or entry.get("token") != token:
            return False
        self.request_watchdogs.pop(session_id, None)
        watchdog_task = entry.get("watchdog_task")
        if not entry.get("timed_out") and watchdog_task and not watchdog_task.done():
            watchdog_task.cancel()
        entry["finish_reason"] = finish_reason
        entry["finished_at"] = time.monotonic()
        return True

    async def _request_timeout_watchdog(
        self,
        event: AstrMessageEvent,
        session_id: str,
        token: int,
        generation: int,
        pipeline_task: asyncio.Task,
        timeout_seconds: int,
    ):
        try:
            await asyncio.sleep(timeout_seconds)
            entry = self.request_watchdogs.get(session_id)
            if (
                not entry
                or entry.get("token") != token
                or entry.get("generation") != generation
                or entry.get("pipeline_task") is not pipeline_task
                or pipeline_task.done()
            ):
                return

            entry["timed_out"] = True
            now = time.monotonic()
            total_elapsed = now - entry["started_at"]
            idle_elapsed = now - entry["last_refreshed_at"]
            logger.error(
                f"【会话请求超时｜强制终止】session={session_id}, token={token}, "
                f"generation={generation}, phase={entry.get('phase')}, "
                f"limit={timeout_seconds}s, idle={idle_elapsed:.1f}s, "
                f"total={total_elapsed:.1f}s；正在取消当前请求并释放会话锁"
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

    def _refresh_request_watchdog(
        self,
        event: AstrMessageEvent,
        phase: str,
    ) -> bool:
        token = event.get_extra("sys_setting_port_request_watchdog_token", None)
        if token is None:
            return False
        session_id = event.unified_msg_origin
        entry = self.request_watchdogs.get(session_id)
        if (
            not entry
            or entry.get("token") != token
            or entry.get("timed_out")
            or entry["pipeline_task"].done()
        ):
            return False

        old_task = entry.get("watchdog_task")
        if old_task and not old_task.done():
            old_task.cancel()
        entry["generation"] += 1
        entry["refresh_count"] += 1
        entry["phase"] = phase
        entry["last_refreshed_at"] = time.monotonic()
        generation = entry["generation"]
        entry["watchdog_task"] = asyncio.create_task(
            self._request_timeout_watchdog(
                event,
                session_id,
                token,
                generation,
                entry["pipeline_task"],
                entry["timeout_seconds"],
            )
        )
        logger.info(
            f"【会话请求看门狗｜已刷新】session={session_id}, token={token}, "
            f"generation={generation}, refresh={entry['refresh_count']}, "
            f"phase={phase}, limit={entry['timeout_seconds']}s"
        )
        return True

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
        started_at = time.monotonic()
        entry = {
            "token": token,
            "pipeline_task": pipeline_task,
            "watchdog_task": None,
            "started_at": started_at,
            "last_refreshed_at": started_at,
            "timeout_seconds": timeout_seconds,
            "generation": 1,
            "refresh_count": 0,
            "phase": "waiting_initial_llm",
            "timed_out": False,
        }
        self.request_watchdogs[session_id] = entry
        event.set_extra("sys_setting_port_request_watchdog_token", token)
        watchdog_task = asyncio.create_task(
            self._request_timeout_watchdog(
                event,
                session_id,
                token,
                entry["generation"],
                pipeline_task,
                timeout_seconds,
            )
        )
        entry["watchdog_task"] = watchdog_task
        pipeline_task.add_done_callback(
            lambda _task, sid=session_id, tok=token: self._finish_request_watchdog(
                sid,
                tok,
                "pipeline_done",
            )
        )
        logger.info(
            f"【会话请求看门狗｜已启动】session={session_id}, token={token}, "
            f"generation=1, limit={timeout_seconds}s, phase=waiting_initial_llm"
        )

    @on_using_llm_tool(priority=maxsize)
    async def refresh_request_watchdog_on_tool_start(
        self,
        event: AstrMessageEvent,
        tool,
        tool_args: dict | None,
    ):
        self._refresh_request_watchdog(event, "tool_started")

    @on_llm_tool_respond(priority=maxsize)
    async def refresh_request_watchdog_on_tool_end(
        self,
        event: AstrMessageEvent,
        tool,
        tool_args: dict | None,
        tool_result,
    ):
        self._refresh_request_watchdog(event, "waiting_next_llm")

    @on_llm_response(priority=maxsize)
    async def finish_request_watchdog_on_llm_response(
        self,
        event: AstrMessageEvent,
        response: LLMResponse,
    ):
        token = event.get_extra("sys_setting_port_request_watchdog_token", None)
        if token is None:
            return
        session_id = event.unified_msg_origin
        entry = self.request_watchdogs.get(session_id)
        if not entry or entry.get("token") != token:
            return
        elapsed = time.monotonic() - entry["started_at"]
        if self._finish_request_watchdog(session_id, token, "llm_response"):
            logger.info(
                f"【会话请求看门狗｜已解除】session={session_id}, token={token}, "
                f"elapsed={elapsed:.1f}s, reason=first_final_llm_response"
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
                        is_gif INTEGER NOT NULL DEFAULT 0,
                        sender_name TEXT NOT NULL DEFAULT '',
                        sender_id TEXT NOT NULL DEFAULT '',
                        created_at REAL NOT NULL,
                        PRIMARY KEY (session_id, message_id, image_key)
                    )
                    """
                )
                cursor = await db.execute("PRAGMA table_info(group_image_captions)")
                caption_columns = {str(row[1]) for row in await cursor.fetchall()}
                if "is_gif" not in caption_columns:
                    await db.execute(
                        "ALTER TABLE group_image_captions "
                        "ADD COLUMN is_gif INTEGER NOT NULL DEFAULT 0"
                    )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_group_image_session_time "
                    "ON group_image_captions(session_id, created_at DESC)"
                )
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS group_image_fingerprints (
                        session_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        position INTEGER NOT NULL,
                        sha256 TEXT NOT NULL,
                        caption TEXT NOT NULL,
                        image_count INTEGER NOT NULL,
                        is_gif INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL,
                        hit_count INTEGER NOT NULL DEFAULT 1,
                        first_seen REAL NOT NULL DEFAULT 0,
                        last_seen REAL NOT NULL DEFAULT 0,
                        PRIMARY KEY (session_id, message_id, position)
                    )
                    """
                )
                cursor = await db.execute("PRAGMA table_info(group_image_fingerprints)")
                fingerprint_columns = {str(row[1]) for row in await cursor.fetchall()}
                for column, definition in (
                    ("is_gif", "INTEGER NOT NULL DEFAULT 0"),
                    ("hit_count", "INTEGER NOT NULL DEFAULT 1"),
                    ("first_seen", "REAL NOT NULL DEFAULT 0"),
                    ("last_seen", "REAL NOT NULL DEFAULT 0"),
                ):
                    if column not in fingerprint_columns:
                        await db.execute(
                            f"ALTER TABLE group_image_fingerprints ADD COLUMN {column} {definition}"
                        )
                await db.execute(
                    "UPDATE group_image_fingerprints SET "
                    "hit_count = CASE WHEN hit_count < 1 THEN 1 ELSE hit_count END, "
                    "first_seen = CASE WHEN first_seen <= 0 THEN created_at ELSE first_seen END, "
                    "last_seen = CASE WHEN last_seen <= 0 THEN created_at ELSE last_seen END"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_group_image_fingerprint_sha_time "
                    "ON group_image_fingerprints(sha256, last_seen DESC)"
                )
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS group_role_snapshots (
                        platform_id TEXT NOT NULL,
                        group_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        display_name TEXT NOT NULL DEFAULT '',
                        role TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (platform_id, group_id, user_id)
                    )
                    """
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_group_role_snapshot_group "
                    "ON group_role_snapshots(platform_id, group_id, role)"
                )
                await db.commit()
            self.caption_db_ready = True

    @staticmethod
    def _unwrap_onebot_list(payload) -> list[dict]:
        if isinstance(payload, dict):
            for key in ("data", "members", "items"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []

    @staticmethod
    def _member_display_name(member: dict) -> str:
        return str(member.get("card") or member.get("nickname") or member.get("name") or member.get("user_id") or "")

    async def _load_group_role_snapshot(self, platform_id: str, group_id: str) -> list[dict]:
        await self._ensure_caption_db()
        async with self.caption_db_lock:
            async with aiosqlite.connect(self.caption_db_path) as db:
                cursor = await db.execute(
                    "SELECT user_id, display_name, role, updated_at "
                    "FROM group_role_snapshots WHERE platform_id = ? AND group_id = ? "
                    "ORDER BY CASE role WHEN 'owner' THEN 0 ELSE 1 END, user_id",
                    (platform_id, group_id),
                )
                rows = await cursor.fetchall()
        return [
            {
                "user_id": str(row[0]),
                "display_name": str(row[1]),
                "role": str(row[2]),
                "updated_at": float(row[3]),
            }
            for row in rows
        ]

    async def _replace_group_role_snapshot(
        self,
        platform_id: str,
        group_id: str,
        members: list[dict],
    ) -> tuple[bool, str]:
        privileged = []
        for member in members:
            role = str(member.get("role") or "").lower()
            user_id = str(member.get("user_id") or member.get("uin") or "").strip()
            if role in {"owner", "admin"} and user_id:
                privileged.append(
                    {
                        "user_id": user_id,
                        "display_name": self._member_display_name(member),
                        "role": role,
                    }
                )
        owners = [item for item in privileged if item["role"] == "owner"]
        if len(owners) != 1:
            return False, f"返回数据中群主数量为 {len(owners)}，保留原快照"

        old_rows = await self._load_group_role_snapshot(platform_id, group_id)
        old_state = {
            (item["user_id"], item["display_name"], item["role"])
            for item in old_rows
        }
        new_state = {
            (item["user_id"], item["display_name"], item["role"])
            for item in privileged
        }
        if old_state == new_state:
            return True, "角色信息没有变化"

        updated_at = time.time()
        async with self.caption_db_lock:
            async with aiosqlite.connect(self.caption_db_path) as db:
                await db.execute("BEGIN")
                await db.execute(
                    "DELETE FROM group_role_snapshots WHERE platform_id = ? AND group_id = ?",
                    (platform_id, group_id),
                )
                await db.executemany(
                    "INSERT INTO group_role_snapshots "
                    "(platform_id, group_id, user_id, display_name, role, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            platform_id,
                            group_id,
                            item["user_id"],
                            item["display_name"],
                            item["role"],
                            updated_at,
                        )
                        for item in privileged
                    ],
                )
                await db.commit()
        return True, f"已保存 1 位群主和 {len(privileged) - 1} 位管理员"

    async def _refresh_group_roles(self, bot, platform_id: str, group_id: str) -> tuple[bool, str]:
        try:
            payload = await asyncio.wait_for(
                bot.call_action("get_group_member_list", group_id=int(group_id)),
                timeout=30.0,
            )
        except Exception as e:
            return False, f"NapCat 查询失败：{type(e).__name__}: {e}"
        members = self._unwrap_onebot_list(payload)
        if not members:
            return False, "NapCat 未返回群成员，保留原快照"
        return await self._replace_group_role_snapshot(platform_id, group_id, members)

    def _whitelist_group_targets(self) -> list[tuple[str, str]]:
        try:
            config = self.context.get_config()
            entries = config.get("platform_settings", {}).get("id_whitelist", []) or []
        except Exception as e:
            logger.warning(f"读取 AstrBot 系统会话白名单失败: {e}")
            return []
        targets = []
        for raw in entries:
            value = str(raw).strip()
            if value.isdigit():
                targets.append(("", value))
                continue
            parts = value.split(":")
            if len(parts) >= 3 and parts[-2] == "GroupMessage" and parts[-1].isdigit():
                targets.append((parts[0], parts[-1]))
        return list(dict.fromkeys(targets))

    def _aiocqhttp_platforms(self):
        platforms = []
        for platform in getattr(self.context.platform_manager, "platform_insts", []):
            try:
                if platform.meta().name == "aiocqhttp":
                    platforms.append(platform)
            except Exception:
                continue
        return platforms

    async def _sync_whitelist_group_roles(self) -> dict[str, int]:
        async with self.group_role_sync_lock:
            return await self._sync_whitelist_group_roles_unlocked()

    async def _sync_whitelist_group_roles_unlocked(self) -> dict[str, int]:
        targets = self._whitelist_group_targets()
        stats = {
            "whitelist": len(targets),
            "platforms": 0,
            "eligible": 0,
            "success": 0,
            "failed": 0,
        }
        if not targets:
            logger.info("【群角色同步】AstrBot 系统白名单中没有群会话，跳过")
            return stats
        platforms = self._aiocqhttp_platforms()
        stats["platforms"] = len(platforms)
        if not platforms:
            logger.warning("【群角色同步】当前没有可用的 aiocqhttp 平台")
            return stats
        for platform in platforms:
            platform_id = str(platform.meta().id)
            try:
                joined_payload = await asyncio.wait_for(
                    platform.bot.call_action("get_group_list"), timeout=30.0
                )
                joined_groups = {
                    str(item.get("group_id"))
                    for item in self._unwrap_onebot_list(joined_payload)
                    if item.get("group_id") is not None
                }
            except Exception as e:
                logger.warning(f"【群角色同步】读取平台群列表失败: platform={platform_id}, error={e}")
                stats["failed"] += 1
                continue
            group_ids = list(
                dict.fromkeys(
                    group_id
                    for platform_hint, group_id in targets
                    if (not platform_hint or platform_hint == platform_id)
                    and group_id in joined_groups
                )
            )
            stats["eligible"] += len(group_ids)
            for index, group_id in enumerate(group_ids):
                ok, detail = await self._refresh_group_roles(platform.bot, platform_id, group_id)
                if ok:
                    stats["success"] += 1
                    logger.info(f"【群角色同步】group={group_id}, {detail}")
                else:
                    stats["failed"] += 1
                    logger.warning(f"【群角色同步】group={group_id}, {detail}")
                if index < len(group_ids) - 1:
                    await asyncio.sleep(0.5)
        logger.info(
            "【群角色同步】本轮完成: "
            f"whitelist={stats['whitelist']}, platforms={stats['platforms']}, "
            f"eligible={stats['eligible']}, success={stats['success']}, "
            f"failed={stats['failed']}"
        )
        return stats

    async def _group_role_sync_loop(self) -> None:
        try:
            while True:
                stats = await self._sync_whitelist_group_roles()
                if stats["whitelist"] == 0 or (
                    stats["eligible"] > 0 and stats["failed"] == 0
                ):
                    break
                logger.warning("【群角色同步】首次同步尚未完成，5 分钟后自动补偿")
                await asyncio.sleep(300)
            interval_days = max(1, int(self.config.get("group_role_sync_interval_days", 7)))
            while True:
                await asyncio.sleep(interval_days * 86400)
                await self._sync_whitelist_group_roles()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"群角色周期同步异常退出: {e}", exc_info=True)

    def _ensure_group_role_sync_task(self) -> None:
        if not self.config.get("enable_group_role_context", True):
            return
        if self.group_role_sync_task and not self.group_role_sync_task.done():
            return
        self.group_role_sync_task = asyncio.create_task(self._group_role_sync_loop())

    @on_astrbot_loaded()
    async def start_group_role_sync(self, *args, **kwargs):
        self._ensure_group_role_sync_task()

    @permission_type(PermissionType.ADMIN)
    @command("同步群角色")
    async def sync_all_group_roles_command(self, event: AstrMessageEvent):
        stats = await self._sync_whitelist_group_roles()
        yield event.plain_result(
            "群角色同步完成："
            f"系统白名单群 {stats['whitelist']} 个，"
            f"当前可同步 {stats['eligible']} 个，"
            f"成功 {stats['success']} 个，失败 {stats['failed']} 个。"
        )

    @llm_tool("sys_refresh_group_roles")
    async def refresh_current_group_roles(self, event: AstrMessageEvent) -> str:
        """重新查询并持久化当前 QQ 群的群主与管理员。当聊天中有人提到群主、管理员或权限发生变化，或者当前角色快照可能过时时使用。只刷新当前会话所在群；查询失败或返回无效数据时保留旧快照。"""
        group_id = str(event.get_group_id() or "").strip()
        if not group_id or event.get_platform_name() != "aiocqhttp":
            return "刷新失败：此工具只能在当前 QQ 群会话中使用。"
        platform_id = str(event.get_platform_id() or "").strip()
        bot = getattr(event, "bot", None)
        if bot is None:
            return "刷新失败：无法取得当前 QQ 平台连接。"
        ok, detail = await self._refresh_group_roles(bot, platform_id, group_id)
        if not ok:
            return f"刷新失败：{detail}"
        rows = await self._load_group_role_snapshot(platform_id, group_id)
        owner = next((item for item in rows if item["role"] == "owner"), None)
        admins = [item for item in rows if item["role"] == "admin"]
        owner_text = f"{owner['display_name']}({owner['user_id']})" if owner else "未知"
        admin_text = "、".join(f"{item['display_name']}({item['user_id']})" for item in admins) or "无"
        return f"当前群角色已核验：{detail}。群主：{owner_text}；管理员：{admin_text}。"

    async def _build_group_role_context(self, event: AstrMessageEvent) -> str:
        if not self.config.get("enable_group_role_context", True):
            return ""
        group_id = str(event.get_group_id() or "").strip()
        platform_id = str(event.get_platform_id() or "").strip()
        if not group_id or event.get_platform_name() != "aiocqhttp":
            return ""
        rows = await self._load_group_role_snapshot(platform_id, group_id)
        if not rows:
            return (
                "<group_role_context>\n"
                f"当前群: {group_id}\n"
                "群角色快照尚未建立。不要猜测任何人的群权限；必要时调用 sys_refresh_group_roles 核验当前群。\n"
                "</group_role_context>"
            )
        owner = next((item for item in rows if item["role"] == "owner"), None)
        admins = [item for item in rows if item["role"] == "admin"]
        role_by_id = {item["user_id"]: item["role"] for item in rows}
        sender_id = str(event.get_sender_id() or "")
        sender_name = str(event.get_sender_name() or sender_id)
        self_id = str(getattr(event.message_obj, "self_id", "") or "")

        def role_label(user_id: str) -> str:
            return {"owner": "群主", "admin": "管理员"}.get(
                role_by_id.get(user_id), "普通群成员"
            )

        owner_text = f"{owner['display_name']}({owner['user_id']})" if owner else "未知"
        admin_text = "、".join(f"{item['display_name']}({item['user_id']})" for item in admins) or "无"
        lines = [
            "<group_role_context>",
            "以下是当前 QQ 群的持久化角色快照，仅用于本轮判断，不是用户原话。",
            f"当前群: {group_id}",
            f"Bot 自身身份: {role_label(self_id)} (QQ: {self_id or '未知'})",
            f"群主: {owner_text}",
            f"管理员: {admin_text}",
        ]
        if not event.get_extra("crossflow_synthetic_event", False):
            lines.append(f"当前发言人: {sender_name}({sender_id})，身份: {role_label(sender_id)}")
        else:
            lines.append("当前事件是跨会话委托，不要把来源请求者误认成目标群的当前发言人。")
        lines.extend(
            [
                "不在上述群主或管理员名单中的群成员，按普通群成员理解。",
                "若对话明确提到群主、管理员或权限刚发生变化，可调用 sys_refresh_group_roles 重新核验；不要仅凭聊天说法自行改写身份。",
                "</group_role_context>",
            ]
        )
        return "\n".join(lines)

    def _caption_cache_ttl_seconds(self) -> int:
        hours = max(0, int(self.config.get("group_visual_cache_ttl_hours", 24)))
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
            "SELECT message_id, caption, is_gif FROM group_image_captions "
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
        for message_id, caption, is_gif in rows:
            if caption:
                caption_text = str(caption)
                if bool(is_gif):
                    caption_text = self._with_gif_human_hint(caption_text)
                result.setdefault(str(message_id), []).append(caption_text)
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
        is_gif: bool = False,
    ):
        valid = [
            (str(key), self._strip_gif_human_hint(str(caption).strip()))
            for key, caption in captions
            if str(caption).strip()
        ]
        if not session_id or not message_id or not valid:
            return
        await self._ensure_caption_db()
        now = time.time()
        ttl_seconds = self._caption_cache_ttl_seconds()
        async with self.caption_db_lock:
            async with aiosqlite.connect(self.caption_db_path) as db:
                await db.execute("PRAGMA busy_timeout=10000")
                await db.executemany(
                    """
                    INSERT INTO group_image_captions
                    (session_id, message_id, image_key, caption, is_gif,
                     sender_name, sender_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, message_id, image_key) DO UPDATE SET
                        caption=excluded.caption,
                        is_gif=excluded.is_gif,
                        sender_name=excluded.sender_name,
                        sender_id=excluded.sender_id,
                        created_at=excluded.created_at
                    """,
                    [
                        (
                            session_id, message_id, key, caption, int(is_gif),
                            sender_name, sender_id, now,
                        )
                        for key, caption in valid
                    ],
                )
                if ttl_seconds > 0:
                    await db.execute(
                        "DELETE FROM group_image_captions WHERE created_at < ?",
                        (now - ttl_seconds,),
                    )
                await db.commit()

    @staticmethod
    def _hash_image_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            while chunk := file.read(65536):
                digest.update(chunk)
        return digest.hexdigest()

    async def _hash_image_paths(self, paths: list[str]) -> list[str]:
        selected_paths = paths[:1] if len(paths) == 1 else paths[:3]
        return [
            await asyncio.to_thread(self._hash_image_file, path)
            for path in selected_paths
        ]

    async def _find_reusable_image_caption(self, hashes: list[str]) -> str:
        if not hashes:
            return ""
        await self._ensure_caption_db()
        unique_hashes = list(dict.fromkeys(hashes))
        placeholders = ",".join("?" for _ in unique_hashes)
        query = (
            "SELECT session_id, message_id, position, sha256, caption, "
            "image_count, is_gif, first_seen, last_seen FROM group_image_fingerprints "
            f"WHERE sha256 IN ({placeholders}) ORDER BY last_seen DESC"
        )
        async with aiosqlite.connect(self.caption_db_path) as db:
            cursor = await db.execute(query, unique_hashes)
            rows = await cursor.fetchall()

        candidates = {}
        for session_id, message_id, position, sha256, caption, image_count, is_gif, first_seen, last_seen in rows:
            key = (str(session_id), str(message_id))
            candidate = candidates.setdefault(
                key,
                {
                    "key": key,
                    "hashes": [],
                    "caption": str(caption),
                    "image_count": int(image_count),
                    "is_gif": bool(is_gif),
                    "first_seen": float(first_seen),
                    "last_seen": float(last_seen),
                },
            )
            candidate["hashes"].append((int(position), str(sha256)))

        current_counter = Counter(hashes)
        matched = None
        for candidate in sorted(
            candidates.values(),
            key=lambda item: item["last_seen"],
            reverse=True,
        ):
            candidate_hashes = [
                sha256 for _, sha256 in sorted(candidate["hashes"])
            ]
            if len(hashes) == 1:
                if candidate["image_count"] == 1 and candidate_hashes == hashes:
                    matched = candidate
                    break
                continue
            overlap = sum((current_counter & Counter(candidate_hashes)).values())
            if candidate["image_count"] >= 2 and overlap >= 2:
                matched = candidate
                break
        if not matched:
            return ""

        now = time.time()
        async with self.caption_db_lock:
            async with aiosqlite.connect(self.caption_db_path) as db:
                await db.execute(
                    "UPDATE group_image_fingerprints SET hit_count = hit_count + 1, "
                    "last_seen = ? WHERE session_id = ? AND message_id = ?",
                    (now, *matched["key"]),
                )
                await db.commit()
        return (
            self._with_gif_human_hint(matched["caption"])
            if matched["is_gif"]
            else matched["caption"]
        )

    async def _replace_image_fingerprint_caption(
        self,
        hashes: list[str],
        caption: str,
        is_gif: bool = False,
    ) -> bool:
        caption = self._strip_gif_human_hint(str(caption).strip())
        if not hashes or not caption:
            return False
        await self._ensure_caption_db()
        unique_hashes = list(dict.fromkeys(hashes))
        placeholders = ",".join("?" for _ in unique_hashes)
        query = (
            "SELECT session_id, message_id, position, sha256, image_count, last_seen "
            "FROM group_image_fingerprints "
            f"WHERE sha256 IN ({placeholders}) ORDER BY last_seen DESC"
        )
        async with self.caption_db_lock:
            async with aiosqlite.connect(self.caption_db_path) as db:
                cursor = await db.execute(query, unique_hashes)
                rows = await cursor.fetchall()
                candidates = {}
                for session_id, message_id, position, sha256, image_count, last_seen in rows:
                    key = (str(session_id), str(message_id))
                    candidate = candidates.setdefault(
                        key,
                        {
                            "key": key,
                            "hashes": [],
                            "image_count": int(image_count),
                            "last_seen": float(last_seen),
                        },
                    )
                    candidate["hashes"].append((int(position), str(sha256)))

                current_counter = Counter(hashes)
                matched = None
                for candidate in sorted(
                    candidates.values(),
                    key=lambda item: item["last_seen"],
                    reverse=True,
                ):
                    candidate_hashes = [
                        sha256 for _, sha256 in sorted(candidate["hashes"])
                    ]
                    if len(hashes) == 1:
                        if candidate["image_count"] == 1 and candidate_hashes == hashes:
                            matched = candidate
                            break
                        continue
                    overlap = sum(
                        (current_counter & Counter(candidate_hashes)).values()
                    )
                    if candidate["image_count"] >= 2 and overlap >= 2:
                        matched = candidate
                        break
                if not matched:
                    return False

                await db.execute(
                    "UPDATE group_image_fingerprints SET caption = ?, is_gif = ?, "
                    "hit_count = hit_count + 1, last_seen = ? "
                    "WHERE session_id = ? AND message_id = ?",
                    (caption, int(is_gif), time.time(), *matched["key"]),
                )
                await db.commit()
        return True

    async def _prune_image_fingerprints(self, db, now: float):
        max_groups = max(1, int(self.config.get("image_fingerprint_cache_size", 100000)))
        prune_groups = max(1, int(self.config.get("image_fingerprint_prune_count", 10000)))
        cursor = await db.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM group_image_fingerprints "
            "GROUP BY session_id, message_id)"
        )
        group_count = int((await cursor.fetchone())[0])
        if group_count <= max_groups:
            return
        delete_count = min(prune_groups, group_count)
        cursor = await db.execute(
            "SELECT session_id, message_id, MAX(hit_count) AS hits, "
            "MIN(first_seen) AS first_seen, MAX(last_seen) AS last_seen "
            "FROM group_image_fingerprints GROUP BY session_id, message_id"
        )
        groups = await cursor.fetchall()
        groups.sort(
            key=lambda row: (
                float(row[2]) / max(1.0, (now - float(row[3])) / 86400.0),
                float(row[4]),
            )
        )
        await db.executemany(
            "DELETE FROM group_image_fingerprints WHERE session_id = ? AND message_id = ?",
            [(str(row[0]), str(row[1])) for row in groups[:delete_count]],
        )
        logger.info(
            f"SHA-256 长期描述库已按热度淘汰: before={group_count}, "
            f"deleted={delete_count}, limit={max_groups}"
        )

    async def _save_image_fingerprints(
        self,
        session_id: str,
        message_id: str,
        hashes: list[str],
        caption: str,
        image_count: int,
        is_gif: bool = False,
    ):
        caption = self._strip_gif_human_hint(str(caption).strip())
        if not session_id or not message_id or not hashes or not caption:
            return
        await self._ensure_caption_db()
        now = time.time()
        async with self.caption_db_lock:
            async with aiosqlite.connect(self.caption_db_path) as db:
                await db.execute("PRAGMA busy_timeout=10000")
                await db.execute(
                    "DELETE FROM group_image_fingerprints "
                    "WHERE session_id = ? AND message_id = ?",
                    (session_id, message_id),
                )
                await db.executemany(
                    """
                    INSERT INTO group_image_fingerprints
                    (session_id, message_id, position, sha256, caption, image_count,
                     is_gif, created_at, hit_count, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    [
                        (
                            session_id,
                            message_id,
                            position,
                            sha256,
                            caption,
                            image_count,
                            int(is_gif),
                            now,
                            now,
                            now,
                        )
                        for position, sha256 in enumerate(hashes)
                    ],
                )
                await self._prune_image_fingerprints(db, now)
                await db.commit()

    @staticmethod
    def _image_cache_key(image: Image, index: int = 0) -> str:
        return str(
            getattr(image, "url", None)
            or getattr(image, "file", None)
            or f"image_{index}"
        )

    async def _resolve_group_image_paths(
        self,
        event: AstrMessageEvent,
        images: list[Image],
    ) -> list[str]:
        paths = []
        for index, image in enumerate(images):
            path = await self._resolve_valid_image_path(
                event,
                image,
                kind="group_silent",
                index=index,
            )
            if path:
                paths.append(path)
        if len(paths) != len(images):
            logger.warning(
                f"群聊图片理解跳过无效图片: total={len(images)}, valid={len(paths)}"
            )
        return paths

    async def _caption_group_images(
        self,
        event: AstrMessageEvent,
        images: list[Image],
        paths: list[str] | None = None,
    ) -> str:
        provider_id = self.config.get("group_visual_provider_id", "")
        fallback_provider_id = self.config.get(
            "group_visual_fallback_provider_id",
            "",
        )
        if not provider_id or not images:
            return ""
        prompt = self.config.get(
            "group_visual_prompt",
            "请用简洁准确的中文描述图片中的主体、动作、场景、文字和重要细节，供群聊上下文理解。只输出图片描述。",
        )
        if len(images) > 1:
            prompt += f"\n本条消息包含 {len(images)} 张图片，请按图片顺序给出一份联合描述并说明它们之间的关系。"
        max_retries = max(
            1,
            int(self.config.get("group_visual_max_retries", 3)),
        )
        timeout_seconds = max(
            1,
            int(self.config.get("group_visual_llm_timeout_seconds", 120)),
        )
        try:
            if paths is None:
                paths = await self._resolve_group_image_paths(event, images)
            if not paths:
                logger.error(
                    f"群聊图片理解已取消: images={len(images)}，没有可发送的有效图片"
                )
                return ""
            visual_paths, gif_converted = self._prepare_visual_image_paths(
                paths,
                bool(self.config.get("enable_gif_frame_staticization", True)),
            )
            visual_prompt = prompt
            if gif_converted:
                visual_prompt = f"{prompt}\n\n{self._gif_human_hint()}"
            caption = await self._try_caption(
                provider_id,
                visual_prompt,
                visual_paths,
                max_retries,
                timeout_seconds,
            )
            if caption:
                return caption
            if fallback_provider_id and fallback_provider_id != provider_id:
                fallback_provider = self.context.get_provider_by_id(
                    fallback_provider_id
                )
                logger.warning(
                    f"群聊图片理解主模型失败，准备调用兜底模型: "
                    f"provider={self._provider_log_name(fallback_provider_id, fallback_provider)}, "
                    f"images={len(paths)}, retries={max_retries}, timeout={timeout_seconds}s"
                )
                fallback_caption = await self._try_caption(
                    fallback_provider_id,
                    visual_prompt,
                    visual_paths,
                    max_retries,
                    timeout_seconds,
                )
                return fallback_caption
            return ""
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                f"群聊图片理解失败: images={len(images)}, error={type(e).__name__}: {e}",
                exc_info=True,
            )
            return ""

    async def _caption_group_image(
        self,
        event: AstrMessageEvent,
        image: Image,
    ) -> str:
        return await self._caption_group_images(event, [image])

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
        image_count = sum(
            1 for segment in (msg.get("message", []) or [])
            if str(segment.get("type", "")) == "image"
        )
        for segment in msg.get("message", []) or []:
            segment_type = str(segment.get("type", ""))
            data = segment.get("data", {}) or {}
            if segment_type == "text":
                text = str(data.get("text", "")).strip()
                if text:
                    parts.append(text)
            elif segment_type == "image":
                if len(captions) == 1 and image_count > 1:
                    if caption_index == 0:
                        parts.append(f"[本消息多图联合描述：{captions[0]}]")
                    else:
                        parts.append("[同组图片]")
                elif caption_index < len(captions):
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
        if existing:
            return
        logger.info(
            f"群聊静默图片理解已触发: group={cache_session_id}, "
            f"message={message_id}, images={len(images)}"
        )
        try:
            paths = await self._resolve_group_image_paths(event, images)
            if not paths:
                logger.error(
                    f"群聊图片理解已取消: images={len(images)}，没有可发送的有效图片"
                )
                return
            hashes = await self._hash_image_paths(paths)
            is_gif = bool(
                self.config.get("enable_gif_frame_staticization", True)
                and any(self._is_gif_path(path) for path in paths)
            )
            caption = await self._find_reusable_image_caption(hashes)
            reused = bool(caption)
            if reused:
                logger.info(
                    f"群聊静默图片描述已按 SHA-256 复用: group={cache_session_id}, "
                    f"message={message_id}, images={len(images)}, hashed={len(hashes)}"
                )
            else:
                caption = await self._caption_group_images(event, images, paths)
            if caption:
                await self._save_message_captions(
                    cache_session_id,
                    message_id,
                    [("message_image_bundle", caption)],
                    event.get_sender_name() or str(event.get_sender_id()),
                    str(event.get_sender_id()),
                    is_gif=is_gif,
                )
                if not reused:
                    await self._save_image_fingerprints(
                        cache_session_id,
                        message_id,
                        hashes,
                        caption,
                        len(images),
                        is_gif=is_gif,
                    )
                logger.info(
                    f"群聊静默图片描述已持久化: group={cache_session_id}, "
                    f"message={message_id}, images={len(images)}, reused={reused}"
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                f"群聊静默图片去重或理解失败: group={cache_session_id}, "
                f"message={message_id}, error={type(e).__name__}: {e}",
                exc_info=True,
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
        quote_images, quote_sources, quote_captions, quote_records = [], [], [], []
        for comp in event.message_obj.message:
            if isinstance(comp, Reply) and comp.chain:
                sender_name = (getattr(comp, "sender_nickname", None) or "未知用户").strip()
                sender_id = str(getattr(comp, "sender_id", None) or "")
                source_label = f"{sender_name}（{sender_id}）" if sender_id and sender_id not in sender_name else sender_name
                message_id = str(comp.id or "")
                original_message_str = str(comp.message_str or "").strip()
                cached_captions = []
                if event.get_group_id() and message_id:
                    cached_captions = await self._get_message_captions(
                        self._group_cache_session_id(event),
                        message_id,
                    )
                    quote_captions.extend(cached_captions)
                new_chain, record_images = [], []
                for component in comp.chain:
                    if isinstance(component, Image):
                        quote_images.append(component)
                        record_images.append(component)
                        quote_sources.append(source_label)
                    else:
                        new_chain.append(component)
                comp.chain = new_chain
                if record_images:
                    if cached_captions:
                        image_text = f"[图片：{'；'.join(cached_captions)}]"
                        comp.message_str = " ".join(
                            part for part in (original_message_str, image_text) if part
                        )
                        comp.text = comp.message_str
                    quote_records.append(
                        {
                            "message_id": message_id,
                            "sender_name": sender_name,
                            "sender_id": sender_id,
                            "source": source_label,
                            "message_str": original_message_str,
                            "images": record_images,
                            "captions": cached_captions,
                        }
                    )
        if quote_images:
            for image in quote_images:
                event.message_obj.message.append(image)
            event.set_extra("sys_setting_port_quote_images", quote_images)
            event.set_extra("sys_setting_port_quote_sources", quote_sources)
            event.set_extra("sys_setting_port_quote_captions", quote_captions)
            event.set_extra("sys_setting_port_quote_records", quote_records)

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
    def _inject_caption_text(
        event: AstrMessageEvent,
        req: ProviderRequest,
        caption_text: str,
        quote_record: dict | None = None,
    ) -> str:
        if event.is_private_chat():
            if req.extra_user_content_parts is None:
                req.extra_user_content_parts = []
            req.extra_user_content_parts.append(TextPart(text=f"\n{caption_text}"))
            return "private_persistent_user_content"
        pending = list(event.get_extra("sys_setting_port_pending_captions", []))
        item = {
            "kind": "quoted" if quote_record else "current",
            "caption": caption_text,
        }
        if quote_record:
            item.update(
                {
                    "message_id": str(quote_record.get("message_id") or ""),
                    "original_sender_name": str(
                        quote_record.get("sender_name") or "未知用户"
                    ),
                    "original_sender_id": str(quote_record.get("sender_id") or ""),
                    "quoting_sender_name": event.get_sender_name()
                    or str(event.get_sender_id()),
                    "quoting_sender_id": str(event.get_sender_id()),
                }
            )
        if item not in pending:
            pending.append(item)
        event.set_extra("sys_setting_port_pending_captions", pending)
        return "group_pending_quoted_context" if quote_record else "group_pending_current_context"

    @staticmethod
    def _find_native_quote_part_index(req: ProviderRequest) -> int:
        for index, part in enumerate(req.extra_user_content_parts or []):
            if isinstance(part, TextPart) and "<Quoted Message>" in part.text:
                return index
        return -1

    @classmethod
    def _replace_quoted_message_text(
        cls,
        req: ProviderRequest,
        quote_record: dict,
        caption: str,
    ) -> bool:
        index = cls._find_native_quote_part_index(req)
        if index < 0:
            return False
        sender_name = str(quote_record.get("sender_name") or "未知用户")
        original_text = str(quote_record.get("message_str") or "").strip()
        image_text = f"[图片：{caption}]"
        quoted_body = " ".join(part for part in (original_text, image_text) if part)
        req.extra_user_content_parts[index] = TextPart(
            text=f"<Quoted Message>\n({sender_name}): {quoted_body}\n</Quoted Message>"
        )
        return True

    @classmethod
    def _mark_quoted_message_images(
        cls,
        req: ProviderRequest,
        quote_record: dict,
        image_count: int,
        is_gif: bool = False,
    ) -> bool:
        index = cls._find_native_quote_part_index(req)
        if index < 0 or image_count <= 0:
            return False
        sender_name = str(quote_record.get("sender_name") or "未知用户")
        original_text = str(quote_record.get("message_str") or "").strip()
        media_kind = "GIF 动图" if is_gif else "图片"
        image_text = (
            f"[该历史消息包含 {image_count} 张{media_kind}；画面已通过本轮标准多模态图片输入附带，"
            f"发送者是 {sender_name}，不是当前引用者"
            + (f"；{cls._gif_human_hint()}" if is_gif else "")
            + "]"
        )
        quoted_body = " ".join(part for part in (original_text, image_text) if part)
        req.extra_user_content_parts[index] = TextPart(
            text=f"<Quoted Message>\n({sender_name}): {quoted_body}\n</Quoted Message>"
        )
        return True

    @staticmethod
    def _inject_temporary_user_context(req: ProviderRequest, text: str) -> int:
        if req.contexts is None:
            req.contexts = []
        temporary_message = Message(role="user", content=text)
        object.__setattr__(temporary_message, "_no_save", True)
        req.contexts.append(temporary_message)
        return len(req.contexts) - 1

    @staticmethod
    def _gif_human_hint() -> str:
        return (
            "因为只识别了其中三帧，若群友说你看的不完整、看错，或问你的问题转述没提到，"
            "你可以说手机网络不好、动图没加载全等理由"
        )

    @classmethod
    def _strip_gif_human_hint(cls, caption: str) -> str:
        clean_caption = str(caption).replace(cls._gif_human_hint(), "").strip()
        if clean_caption.startswith("[GIF 动图：") and clean_caption.endswith("]"):
            clean_caption = clean_caption[8:-1].strip()
        return clean_caption.rstrip("；。 ")

    @classmethod
    def _with_gif_human_hint(cls, caption: str) -> str:
        clean_caption = cls._strip_gif_human_hint(caption)
        if clean_caption.startswith("[GIF 动图：") and clean_caption.endswith("]"):
            clean_caption = clean_caption[8:-1].strip()
        return f"[GIF 动图：{clean_caption}；{cls._gif_human_hint()}]"

    @classmethod
    def _is_gif_path(cls, path: str) -> bool:
        try:
            with open(path, "rb") as file:
                return file.read(4).startswith(b"GIF8")
        except Exception:
            return False

    @classmethod
    def _gif_frame_payloads(cls, path: str) -> list[str]:
        try:
            from PIL import Image as PILImage

            with PILImage.open(path) as image:
                frame_count = max(1, int(getattr(image, "n_frames", 1)))
                indexes = list(dict.fromkeys((0, frame_count // 2, frame_count - 1)))
                payloads = []
                seen = set()
                for index in indexes:
                    image.seek(index)
                    frame = image.convert("RGB")
                    output = io.BytesIO()
                    frame.save(output, format="JPEG", quality=85, optimize=True)
                    raw = output.getvalue()
                    fingerprint = hashlib.sha256(raw).digest()
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    payloads.append("base64://" + base64.b64encode(raw).decode("ascii"))
                return payloads
        except Exception as error:
            logger.warning(
                f"GIF 静态化失败，将回退原图: path={path}, "
                f"error={type(error).__name__}: {error}"
            )
            return []

    @classmethod
    def _prepare_visual_image_paths(
        cls,
        paths: list[str],
        enabled: bool,
    ) -> tuple[list[str], bool]:
        if not enabled:
            return list(paths), False
        prepared = []
        converted = False
        for path in paths:
            if cls._is_gif_path(path):
                frames = cls._gif_frame_payloads(path)
                if frames:
                    prepared.extend(frames)
                    converted = True
                    continue
            prepared.append(path)
        return prepared, converted

    @staticmethod
    def _inspect_image_path(path: str) -> tuple[bool, str]:
        try:
            if path.startswith("base64://"):
                raw = base64.b64decode(path[9:], validate=True)
                if raw.startswith(b"\xFF\xD8\xFF"):
                    return True, f"jpeg-base64(size={len(raw)})"
                return False, f"base64-unknown(size={len(raw)})"
            if not os.path.exists(path):
                return False, "不存在"
            size = os.path.getsize(path)
            if size < 16:
                return False, f"空/过小文件(size={size})"
            with open(path, "rb") as file:
                header = file.read(16)
            signatures = {
                b"\xFF\xD8\xFF": "jpeg",
                b"\x89PNG\r\n\x1a\n": "png",
                b"GIF8": "gif",
                b"BM": "bmp",
                b"II*\x00": "tiff",
                b"MM\x00*": "tiff",
            }
            kind = next(
                (name for signature, name in signatures.items() if header.startswith(signature)),
                "",
            )
            if not kind and header.startswith(b"RIFF") and header[8:12] == b"WEBP":
                kind = "webp"
            if not kind:
                return False, f"unknown(size={size})"
            return True, f"{kind}(size={size})"
        except Exception as e:
            return False, f"检查失败({type(e).__name__}: {e})"

    @classmethod
    def _describe_image_path(cls, path: str) -> str:
        return cls._inspect_image_path(path)[1]

    async def _resolve_valid_image_path(
        self,
        event: AstrMessageEvent,
        image: Image,
        prepared_path: str = "",
        kind: str = "image",
        index: int = 0,
    ) -> str:
        candidates = []
        if prepared_path:
            candidates.append(("astrbot_prepared", prepared_path))
        component_path = str(getattr(image, "path", "") or "")
        if component_path and component_path != prepared_path:
            candidates.append(("component_path", component_path))

        for source, path in candidates:
            valid, diagnostic = self._inspect_image_path(path)
            if valid:
                logger.info(
                    f"【图片转述｜载荷就绪】kind={kind}, source={source}, "
                    f"index={index}, payload={diagnostic}"
                )
                return path
            logger.warning(
                f"【图片转述｜候选载荷无效】kind={kind}, source={source}, "
                f"index={index}, payload={diagnostic}"
            )

        try:
            converted_path = await image.convert_to_file_path()
            valid, diagnostic = self._inspect_image_path(converted_path)
            if valid:
                logger.info(
                    f"【图片转述｜载荷就绪】kind={kind}, source=component_convert, "
                    f"index={index}, payload={diagnostic}"
                )
                return converted_path
            logger.warning(
                f"【图片转述｜组件载荷无效】kind={kind}, index={index}, "
                f"payload={diagnostic}；正在请求 NapCat get_image"
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                f"【图片转述｜组件转换失败】kind={kind}, index={index}, "
                f"error={type(e).__name__}: {e}；正在请求 NapCat get_image"
            )

        bot = getattr(event, "bot", None)
        api = getattr(bot, "api", bot)
        file_id = str(getattr(image, "file", "") or "")
        if api and hasattr(api, "call_action") and file_id:
            try:
                result = await asyncio.wait_for(
                    api.call_action("get_image", file=file_id),
                    timeout=15.0,
                )
                if isinstance(result, dict):
                    protocol_url = str(result.get("url") or "")
                    if protocol_url.startswith(("http://", "https://", "base64://")):
                        try:
                            protocol_path = await Image(file=protocol_url).convert_to_file_path()
                            valid, diagnostic = self._inspect_image_path(protocol_path)
                            if valid:
                                logger.info(
                                    f"【图片转述｜载荷就绪】kind={kind}, "
                                    f"source=napcat_get_image_url, index={index}, "
                                    f"payload={diagnostic}"
                                )
                                return protocol_path
                            logger.warning(
                                f"【图片转述｜NapCat URL 载荷无效】kind={kind}, "
                                f"index={index}, payload={diagnostic}"
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logger.warning(
                                f"【图片转述｜NapCat URL 获取失败】kind={kind}, "
                                f"index={index}, error={type(e).__name__}: {e}"
                            )

                    for field in ("file", "path"):
                        protocol_path = str(result.get(field) or "")
                        if not protocol_path:
                            continue
                        valid, diagnostic = self._inspect_image_path(protocol_path)
                        if valid:
                            logger.info(
                                f"【图片转述｜载荷就绪】kind={kind}, "
                                f"source=napcat_get_image_{field}, index={index}, "
                                f"payload={diagnostic}"
                            )
                            return protocol_path

                    logger.error(
                        f"【图片转述｜NapCat 载荷无效】kind={kind}, index={index}, "
                        f"fields={sorted(result.keys())}, has_url={bool(protocol_url)}"
                    )
                else:
                    logger.error(
                        f"【图片转述｜NapCat 载荷无效】kind={kind}, index={index}, "
                        f"result_type={type(result).__name__}"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(
                    f"【图片转述｜NapCat get_image 失败】kind={kind}, index={index}, "
                    f"error={type(e).__name__}: {e}",
                    exc_info=True,
                )

        logger.error(
            f"【图片转述｜图片载荷无效】kind={kind}, index={index}；"
            "所有来源均无有效图片，已阻止向 Provider 发送"
        )
        return ""

    async def _try_caption(
        self,
        provider_id: str,
        prompt: str,
        image_urls: list,
        max_retries: int,
        timeout_seconds: int | None = None,
    ) -> str:
        invalid_payloads = [
            self._describe_image_path(path)
            for path in image_urls
            if not self._inspect_image_path(path)[0]
        ]
        if not image_urls or invalid_payloads:
            logger.error(
                f"【图片转述｜调用已阻止】provider={provider_id}, images={len(image_urls)}, "
                f"invalid={invalid_payloads or ['empty']}"
            )
            return ""
        prov = self.context.get_provider_by_id(provider_id)
        if not prov:
            logger.error(f"【图片转述｜调用失败】未找到 Provider: {provider_id}")
            return ""
        timeout_seconds = max(
            1,
            int(
                timeout_seconds
                if timeout_seconds is not None
                else self.config.get("caption_llm_timeout_seconds", 90)
            ),
        )
        judge_timeout_seconds = max(
            1,
            int(self.config.get("caption_judge_timeout_seconds", 30)),
        )
        structured_enabled = self.config.get("enable_caption_structured", True)
        judge_enabled, judge_provider_id, judge_prompt_tmpl = self.config.get("enable_caption_judge", False), self.config.get("caption_judge_provider_id", ""), self.config.get("caption_judge_prompt", "")
        if structured_enabled: prompt += "\n请务必将最终的图片描述内容包裹在 <caption_result> 标签中。如果无法描述，请输出 <error>原因</error>。"
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"【图片转述｜调用 Provider】provider={self._provider_log_name(provider_id, prov)}, "
                    f"attempt={attempt + 1}/{max_retries}, timeout={timeout_seconds}s, "
                    f"images={len(image_urls)}, "
                    f"payload={[self._describe_image_path(path) for path in image_urls]}"
                )
                resp = await asyncio.wait_for(
                    prov.text_chat(
                        system_prompt=prompt,
                        prompt="[图片]",
                        image_urls=image_urls,
                    ),
                    timeout=timeout_seconds,
                )
                if not resp or not resp.completion_text:
                    logger.warning(
                        f"【图片转述｜Provider 空响应】provider={self._provider_log_name(provider_id, prov)}, "
                        f"attempt={attempt + 1}/{max_retries}"
                    )
                    continue
                raw_text = resp.completion_text.strip()
                caption = raw_text
                if structured_enabled:
                    import re
                    if "<error>" in raw_text:
                        logger.warning(
                            f"【图片转述｜Provider 返回错误标签】provider="
                            f"{self._provider_log_name(provider_id, prov)}, response={raw_text[:300]}"
                        )
                        continue
                    match = re.search(r"<caption_result>(.*?)</caption_result>", raw_text, re.DOTALL)
                    if match:
                        caption = match.group(1).strip()
                    else:
                        logger.warning(
                            f"【图片转述｜结构化标签缺失】provider="
                            f"{self._provider_log_name(provider_id, prov)}，将直接使用原始响应"
                        )
                if judge_enabled and judge_provider_id and judge_prompt_tmpl:
                    judge_prov = self.context.get_provider_by_id(judge_provider_id)
                    if judge_prov:
                        logger.info(
                            f"【图片转述｜调用质量判定】provider="
                            f"{self._provider_log_name(judge_provider_id, judge_prov)}, "
                            f"timeout={judge_timeout_seconds}s"
                        )
                        j_resp = await asyncio.wait_for(
                            judge_prov.text_chat(
                                prompt=judge_prompt_tmpl.replace("{{caption}}", caption)
                            ),
                            timeout=judge_timeout_seconds,
                        )
                        if j_resp and j_resp.completion_text and "否" in j_resp.completion_text.strip():
                            logger.warning(
                                f"【图片转述｜判定未通过】provider="
                                f"{self._provider_log_name(provider_id, prov)}, attempt={attempt + 1}/{max_retries}"
                            )
                            continue
                if caption:
                    return caption
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(
                    f"【图片转述｜Provider 调用异常】provider={self._provider_log_name(provider_id, prov)}, "
                    f"attempt={attempt + 1}/{max_retries}, error={type(e).__name__}: {e}",
                    exc_info=True,
                )
            if attempt < max_retries - 1:
                await asyncio.sleep(1.5)
        return ""

    @on_llm_request(priority=-maxsize)
    async def inject_group_context_finally(self, event: AstrMessageEvent, req: ProviderRequest):
        if event.is_private_chat():
            return

        group_id = str(event.get_group_id() or "")
        try:
            role_context = await self._build_group_role_context(event)
        except Exception as e:
            role_context = ""
            logger.error(f"读取群角色快照失败: group={group_id}, error={e}")
        if role_context:
            if req.contexts is None:
                req.contexts = []
            temporary_message = Message(role="user", content=role_context)
            object.__setattr__(temporary_message, "_no_save", True)
            req.contexts.append(temporary_message)

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
        for item in pending_captions:
            if isinstance(item, str):
                item = {"kind": "current", "caption": item}
            caption_text = str(item.get("caption") or "")
            if not caption_text:
                continue
            if item.get("kind") == "quoted":
                context_text = (
                    "<quoted_image_ownership>\n"
                    f"<original_message_id>{item.get('message_id') or '未知'}</original_message_id>\n"
                    f"<original_sender_name>{item.get('original_sender_name') or '未知用户'}</original_sender_name>\n"
                    f"<original_sender_id>{item.get('original_sender_id') or '未知'}</original_sender_id>\n"
                    f"<quoting_sender_name>{item.get('quoting_sender_name') or '未知用户'}</quoting_sender_name>\n"
                    f"<quoting_sender_id>{item.get('quoting_sender_id') or '未知'}</quoting_sender_id>\n"
                    "<ownership_rule>随本轮请求附带的引用原图属于 original_sender；quoting_sender 只引用了该历史消息，并非图片发送者。</ownership_rule>\n"
                    "</quoted_image_ownership>"
                )
                context_kind = "quoted_ownership"
            else:
                context_text = (
                    f"<current_image_caption>\n{caption_text}\n</current_image_caption>"
                )
                context_kind = "current"
            position = self._inject_temporary_user_context(req, context_text)
            logger.info(
                f"【图片转述｜最终注入】group={group_id}, kind={context_kind}, "
                f"mode=temporary_context, position={position}, chars={len(caption_text)}"
            )

    @on_llm_request()
    async def on_image_caption_req(self, event: AstrMessageEvent, req: ProviderRequest):
        caption_provider_id = self.config.get("caption_provider_id", "")
        fallback_provider_id = self.config.get("fallback_provider_id", "")
        target_models = self.config.get("target_models", [])
        caption_prompt = self.config.get("caption_prompt", "请详细描述这张图片的内容，以便纯文本模型能够理解。")
        max_retries = int(self.config.get("max_retries", 3))
        caption_timeout_seconds = max(
            1,
            int(self.config.get("caption_llm_timeout_seconds", 90)),
        )
        gif_staticization_enabled = bool(
            self.config.get("enable_gif_frame_staticization", True)
        )
        curr_prov = self.context.get_using_provider(event.unified_msg_origin)
        is_target_text_model, matched_keyword, model_candidates = self._match_target_model(req, curr_prov, target_models)

        quote_sources = event.get_extra("sys_setting_port_quote_sources", [])
        quote_captions = list(event.get_extra("sys_setting_port_quote_captions", []))
        quote_images = event.get_extra("sys_setting_port_quote_images", [])
        quote_records = list(event.get_extra("sys_setting_port_quote_records", []))
        quote_image_ids = {id(image) for image in quote_images}
        quote_paths, current_paths, resolved_paths = [], [], {}

        import re
        prepared_paths = list(req.image_urls or [])
        if not prepared_paths:
            for part in req.extra_user_content_parts or []:
                if not isinstance(part, TextPart):
                    continue
                match = re.search(r"\[Image Attachment: path (.*?)\]", part.text)
                if match:
                    prepared_paths.append(match.group(1).strip())

        image_components = []
        seen_image_ids = set()
        for component in event.message_obj.message:
            if isinstance(component, Image) and id(component) not in seen_image_ids:
                seen_image_ids.add(id(component))
                image_components.append(component)

        for index, component in enumerate(image_components):
            image_kind = "quoted" if id(component) in quote_image_ids else "current"
            prepared_path = prepared_paths[index] if index < len(prepared_paths) else ""
            path = await self._resolve_valid_image_path(
                event,
                component,
                prepared_path=prepared_path,
                kind=image_kind,
                index=index,
            )
            if not path:
                continue
            resolved_paths[id(component)] = path
            target_paths = quote_paths if id(component) in quote_image_ids else current_paths
            if path not in target_paths:
                target_paths.append(path)

        all_paths = [*current_paths, *quote_paths]
        if not all_paths:
            return
        visual_all_paths, request_has_gif = self._prepare_visual_image_paths(
            all_paths,
            gif_staticization_enabled,
        )
        req.image_urls = visual_all_paths
        current_has_gif = gif_staticization_enabled and any(
            self._is_gif_path(path) for path in current_paths
        )
        if current_has_gif and not is_target_text_model:
            req.extra_user_content_parts.append(
                TextPart(
                    text=f"[当前用户消息包含 GIF 动图；{self._gif_human_hint()}]"
                )
            )
        source_text = "；".join(dict.fromkeys(quote_sources)) or "原发送者"
        inspect_keywords = self.config.get("quote_image_inspect_keywords", ["仔细看", "看原图", "看细节", "重新看", "重看", "再看"])
        wants_original = bool(quote_paths) and not is_target_text_model and any(
            keyword and keyword in (req.prompt or "") for keyword in inspect_keywords
        )

        if not is_target_text_model:
            if quote_paths and not wants_original and not quote_captions and self._visual_group_enabled(event) and self.config.get("group_visual_provider_id"):
                for image in quote_images:
                    caption = await self._caption_group_image(event, image)
                    if caption:
                        quote_captions.append(caption)
            if quote_paths and quote_captions and not wants_original:
                req.image_urls, _ = self._prepare_visual_image_paths(
                    current_paths,
                    gif_staticization_enabled,
                )
                logger.info(
                    f"【图片转述｜成功】provider=群聊图片描述缓存, "
                    f"mode=quoted_message_native_cache, "
                    f"chars={len('；'.join(quote_captions))}, images={len(quote_paths)}"
                )
            elif quote_paths:
                req.image_urls = visual_all_paths
                marked_images = 0
                for record in quote_records:
                    record_paths = [
                        resolved_paths[id(image)]
                        for image in record.get("images", [])
                        if id(image) in resolved_paths
                    ]
                    record_image_count = len(record_paths)
                    record_has_gif = gif_staticization_enabled and any(
                        self._is_gif_path(path) for path in record_paths
                    )
                    if self._mark_quoted_message_images(
                        req,
                        record,
                        record_image_count,
                        is_gif=record_has_gif,
                    ):
                        marked_images += record_image_count
                if marked_images < len(quote_paths):
                    for record in quote_records:
                        self._inject_caption_text(event, req, "原图直通", record)
                logger.info(
                    f"【引用图片｜多模态直通】引用归属已写入原生引用结构，"
                    f"原图已保留在标准图片通道: quote_images={len(quote_paths)}, "
                    f"current_images={len(current_paths)}, request_images={len(req.image_urls)}, "
                    f"ownership_marked={marked_images}, gif_staticized={request_has_gif}"
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

        async def caption_batch(
            paths: list[str],
            kind: str,
            source: str = "",
            quote_record: dict | None = None,
        ):
            if not paths:
                return
            hashes = await self._hash_image_paths(paths)
            has_gif = gif_staticization_enabled and any(
                self._is_gif_path(path) for path in paths
            )
            force_reinspect = kind == "quoted_reinspect"
            caption = (
                ""
                if force_reinspect
                else await self._find_reusable_image_caption(hashes)
            )
            reused = bool(caption)
            caption_provider = self.context.get_provider_by_id(caption_provider_id)
            if reused:
                logger.info(
                    f"【图片转述｜SHA-256 复用】kind={kind}, images={len(paths)}, "
                    f"hashed={len(hashes)}"
                )
            else:
                visual_paths, gif_converted = self._prepare_visual_image_paths(
                    paths,
                    gif_staticization_enabled,
                )
                visual_prompt = caption_prompt
                if gif_converted:
                    visual_prompt = f"{caption_prompt}\n\n{self._gif_human_hint()}"
                logger.info(
                    f"【图片转述｜触发】kind={kind}, 目标关键词={matched_keyword}, "
                    f"current_models={model_candidates}, caption_provider="
                    f"{self._provider_log_name(caption_provider_id, caption_provider)}, "
                    f"images={len(paths)}, provider_images={len(visual_paths)}, "
                    f"gif_staticized={gif_converted}"
                )
                caption = await self._try_caption(
                    caption_provider_id,
                    visual_prompt,
                    visual_paths,
                    max_retries,
                    caption_timeout_seconds,
                )
            used_provider_id, used_provider = caption_provider_id, caption_provider
            if not caption and fallback_provider_id:
                if reused:
                    visual_paths, gif_converted = self._prepare_visual_image_paths(
                        paths,
                        gif_staticization_enabled,
                    )
                    visual_prompt = caption_prompt
                    if gif_converted:
                        visual_prompt = f"{caption_prompt}\n\n{self._gif_human_hint()}"
                fallback_provider = self.context.get_provider_by_id(fallback_provider_id)
                logger.warning(
                    f"【图片转述｜主模型失败】kind={kind}, 准备调用兜底模型 "
                    f"{self._provider_log_name(fallback_provider_id, fallback_provider)}"
                )
                caption = await self._try_caption(
                    fallback_provider_id,
                    visual_prompt,
                    visual_paths,
                    max_retries,
                    caption_timeout_seconds,
                )
                used_provider_id, used_provider = fallback_provider_id, fallback_provider
            if not caption:
                logger.error(
                    f"【图片转述｜失败】kind={kind}, 所有转述模型均未返回有效描述；"
                    f"target={matched_keyword}, images={len(paths)}"
                )
                return
            caption_text = (
                self._with_gif_human_hint(caption)
                if has_gif
                else caption
            )
            if quote_record and not event.is_private_chat():
                replaced = self._replace_quoted_message_text(
                    req,
                    quote_record,
                    caption_text,
                )
                mode = (
                    "quoted_message_replaced"
                    if replaced
                    else self._inject_caption_text(event, req, caption_text, quote_record)
                )
            else:
                mode = self._inject_caption_text(event, req, caption_text)
            cache_session_id = self._group_cache_session_id(event)
            message_id = ""
            sender_name = ""
            sender_id = ""
            image_key = ""
            if quote_record and event.get_group_id() and quote_record.get("message_id"):
                message_id = str(quote_record["message_id"])
                sender_name = str(quote_record.get("sender_name") or "未知用户")
                sender_id = str(quote_record.get("sender_id") or "")
                image_key = "text_model_quote_bundle"
            elif event.get_group_id() and kind == "current":
                message_id = str(event.message_obj.message_id or "")
                sender_name = event.get_sender_name() or str(event.get_sender_id())
                sender_id = str(event.get_sender_id())
                image_key = "text_model_current_bundle"
            if message_id:
                await self._save_message_captions(
                    cache_session_id,
                    message_id,
                    [(image_key, caption)],
                    sender_name,
                    sender_id,
                    is_gif=has_gif,
                )
                if hashes and not reused:
                    fingerprint_replaced = False
                    if force_reinspect:
                        fingerprint_replaced = await self._replace_image_fingerprint_caption(
                            hashes,
                            caption,
                            is_gif=has_gif,
                        )
                    if not fingerprint_replaced:
                        await self._save_image_fingerprints(
                            cache_session_id,
                            message_id,
                            hashes,
                            caption,
                            len(paths),
                            is_gif=has_gif,
                        )
                logger.info(
                    f"【图片转述｜消息描述已持久化】group={cache_session_id}, "
                    f"message={message_id}, kind={kind}, images={len(paths)}, "
                    f"reused={reused}"
                )
            logger.info(
                f"【图片转述｜成功】kind={kind}, provider="
                f"{self._provider_log_name(used_provider_id, used_provider)}, "
                f"mode={mode}, chars={len(caption)}, images={len(paths)}"
            )

        force_quote_reinspect = bool(quote_paths) and any(
            keyword and keyword in (req.prompt or "")
            for keyword in inspect_keywords
        )
        if quote_records:
            for record in quote_records:
                record_paths = list(
                    dict.fromkeys(
                        resolved_paths[id(image)]
                        for image in record.get("images", [])
                        if id(image) in resolved_paths
                    )
                )
                record_captions = list(record.get("captions", []))
                if record_captions and not force_quote_reinspect:
                    cached_text = f"{'；'.join(record_captions)}"
                    mode = "quoted_message_native_cache"
                    logger.info(
                        f"【图片转述｜成功】kind=quoted_cached, provider=群聊图片描述缓存, "
                        f"mode={mode}, chars={len('；'.join(record_captions))}, "
                        f"images={len(record_paths)}"
                    )
                    continue
                quote_kind = "quoted_reinspect" if force_quote_reinspect else "quoted_uncached"
                await caption_batch(
                    record_paths,
                    quote_kind,
                    str(record.get("source") or "原发送者"),
                    record,
                )
        elif quote_captions and not force_quote_reinspect:
            cached_text = f"{'；'.join(quote_captions)}"
            mode = self._inject_caption_text(event, req, cached_text)
            logger.info(
                f"【图片转述｜成功】kind=quoted_cached, provider=群聊图片描述缓存, "
                f"mode={mode}, chars={len('；'.join(quote_captions))}, images={len(quote_paths)}"
            )
        else:
            quote_kind = "quoted_reinspect" if force_quote_reinspect else "quoted_uncached"
            await caption_batch(quote_paths, quote_kind, source_text)
        await caption_batch(current_paths, "current", "")

"""
CarFin MCP Tool: Context Sync
에이전트 간 컨텍스트 동기화 및 세션 상태 관리
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import redis
import pickle

logger = logging.getLogger("CarFin-MCP.ContextSync")

class ContextSyncError(Exception):
    """컨텍스트 동기화 관련 에러"""
    pass

@dataclass
class SessionContext:
    session_id: str
    user_id: str
    created_at: datetime
    last_updated: datetime
    participating_agents: Set[str]
    shared_variables: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    current_task: Optional[str]
    task_progress: Dict[str, Any]

@dataclass
class AgentState:
    agent_id: str
    session_id: str
    last_activity: datetime
    current_operation: Optional[str]
    local_context: Dict[str, Any]
    pending_messages: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]

class ContextSyncTool:
    """에이전트 간 컨텍스트 동기화 도구"""

    def __init__(self):
        # Redis 연결 (실제 환경에서는 Redis 사용)
        self.use_redis = False  # 개발 환경에서는 메모리 기반
        self.redis_client = None

        if self.use_redis:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                logger.info("✅ Redis 연결 성공")
            except:
                self.use_redis = False
                logger.warning("⚠️ Redis 연결 실패, 메모리 기반 동작")

        # 메모리 기반 저장소
        self.sessions = {}  # session_id -> SessionContext
        self.agent_states = {}  # agent_id -> AgentState
        self.sync_locks = {}  # session_id -> asyncio.Lock
        self.event_subscribers = defaultdict(list)  # event_type -> [callback]

        # 동기화 설정
        self.sync_config = {
            "session_timeout": 3600,  # 1시간
            "agent_timeout": 300,     # 5분
            "max_history_size": 1000,
            "sync_interval": 1.0,     # 1초마다 동기화
            "auto_cleanup": True
        }

        # 이벤트 시스템
        self.event_queue = asyncio.Queue()
        self.event_handlers = {
            "agent_join": self._handle_agent_join,
            "agent_leave": self._handle_agent_leave,
            "context_update": self._handle_context_update,
            "message_broadcast": self._handle_message_broadcast,
            "task_progress": self._handle_task_progress
        }

        # 백그라운드 작업 시작
        asyncio.create_task(self._background_sync_worker())
        asyncio.create_task(self._cleanup_worker())

        logger.info("✅ ContextSync Tool 초기화 완료")

    async def execute(self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        컨텍스트 동기화 실행

        params:
            action: 'create_session' | 'join_session' | 'update_context' | 'sync_state' | 'broadcast'
            session_id: 세션 ID
            agent_id: 에이전트 ID
            user_id: 사용자 ID
            context_data: 동기화할 컨텍스트 데이터
            message: 브로드캐스트 메시지
        """
        try:
            action = params.get("action")

            if action == "create_session":
                return await self._create_session(params)
            elif action == "join_session":
                return await self._join_session(params)
            elif action == "leave_session":
                return await self._leave_session(params)
            elif action == "update_context":
                return await self._update_context(params)
            elif action == "sync_state":
                return await self._sync_agent_state(params)
            elif action == "broadcast":
                return await self._broadcast_to_session(params)
            elif action == "get_session_info":
                return await self._get_session_info(params)
            elif action == "get_agent_states":
                return await self._get_agent_states(params)
            else:
                raise ContextSyncError(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"❌ 컨텍스트 동기화 실행 실패: {e}")
            raise ContextSyncError(f"Context sync execution failed: {e}")

    async def _create_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """새 세션 생성"""
        session_id = params.get("session_id") or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        user_id = params.get("user_id", "anonymous")
        creator_agent = params.get("agent_id", "system")

        # 기존 세션 확인
        if session_id in self.sessions:
            raise ContextSyncError(f"Session {session_id} already exists")

        # 새 세션 컨텍스트 생성
        session_context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            participating_agents={creator_agent},
            shared_variables={},
            conversation_history=[],
            user_preferences=params.get("user_preferences", {}),
            current_task=None,
            task_progress={}
        )

        # 동기화 락 생성
        self.sync_locks[session_id] = asyncio.Lock()

        # 세션 저장
        await self._save_session(session_context)

        # 이벤트 발생
        await self._emit_event("session_created", {
            "session_id": session_id,
            "user_id": user_id,
            "creator_agent": creator_agent
        })

        logger.info(f"🎯 세션 생성: {session_id} by {creator_agent}")

        return {
            "success": True,
            "session_id": session_id,
            "session_context": asdict(session_context),
            "created_at": session_context.created_at.isoformat()
        }

    async def _join_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """세션 참여"""
        session_id = params.get("session_id")
        agent_id = params.get("agent_id")

        if not session_id or not agent_id:
            raise ContextSyncError("Missing session_id or agent_id")

        # 세션 존재 확인
        session_context = await self._load_session(session_id)
        if not session_context:
            raise ContextSyncError(f"Session {session_id} not found")

        # 동기화 락 획득
        async with self.sync_locks[session_id]:
            # 에이전트 추가
            session_context.participating_agents.add(agent_id)
            session_context.last_updated = datetime.now()

            # 에이전트 상태 초기화
            agent_state = AgentState(
                agent_id=agent_id,
                session_id=session_id,
                last_activity=datetime.now(),
                current_operation=None,
                local_context={},
                pending_messages=[],
                performance_metrics={}
            )

            self.agent_states[agent_id] = agent_state

            # 세션 업데이트
            await self._save_session(session_context)

        # 참여 이벤트 발생
        await self._emit_event("agent_join", {
            "session_id": session_id,
            "agent_id": agent_id,
            "participating_agents": list(session_context.participating_agents)
        })

        logger.info(f"👥 에이전트 세션 참여: {agent_id} → {session_id}")

        return {
            "success": True,
            "session_id": session_id,
            "agent_id": agent_id,
            "session_context": asdict(session_context),
            "participating_agents": list(session_context.participating_agents)
        }

    async def _leave_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """세션 떠나기"""
        session_id = params.get("session_id")
        agent_id = params.get("agent_id")

        if not session_id or not agent_id:
            raise ContextSyncError("Missing session_id or agent_id")

        # 세션 존재 확인
        session_context = await self._load_session(session_id)
        if not session_context:
            return {"success": True, "message": "Session already removed"}

        # 동기화 락 획득
        async with self.sync_locks[session_id]:
            # 에이전트 제거
            session_context.participating_agents.discard(agent_id)
            session_context.last_updated = datetime.now()

            # 에이전트 상태 제거
            if agent_id in self.agent_states:
                del self.agent_states[agent_id]

            # 세션 업데이트 또는 제거
            if session_context.participating_agents:
                await self._save_session(session_context)
            else:
                # 마지막 에이전트가 떠나면 세션 제거
                await self._remove_session(session_id)

        # 떠나기 이벤트 발생
        await self._emit_event("agent_leave", {
            "session_id": session_id,
            "agent_id": agent_id,
            "remaining_agents": list(session_context.participating_agents)
        })

        logger.info(f"👋 에이전트 세션 떠남: {agent_id} ← {session_id}")

        return {
            "success": True,
            "session_id": session_id,
            "agent_id": agent_id,
            "remaining_agents": list(session_context.participating_agents)
        }

    async def _update_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """컨텍스트 업데이트"""
        session_id = params.get("session_id")
        agent_id = params.get("agent_id")
        context_updates = params.get("context_data", {})

        if not session_id or not agent_id:
            raise ContextSyncError("Missing session_id or agent_id")

        # 세션 존재 확인
        session_context = await self._load_session(session_id)
        if not session_context:
            raise ContextSyncError(f"Session {session_id} not found")

        # 동기화 락 획득
        async with self.sync_locks[session_id]:
            # 공유 변수 업데이트
            if "shared_variables" in context_updates:
                session_context.shared_variables.update(context_updates["shared_variables"])

            # 대화 히스토리 추가
            if "conversation_entry" in context_updates:
                entry = context_updates["conversation_entry"]
                entry["timestamp"] = datetime.now().isoformat()
                entry["agent_id"] = agent_id
                session_context.conversation_history.append(entry)

                # 히스토리 크기 제한
                if len(session_context.conversation_history) > self.sync_config["max_history_size"]:
                    session_context.conversation_history = session_context.conversation_history[-self.sync_config["max_history_size"]:]

            # 사용자 선호도 업데이트
            if "user_preferences" in context_updates:
                session_context.user_preferences.update(context_updates["user_preferences"])

            # 현재 태스크 업데이트
            if "current_task" in context_updates:
                session_context.current_task = context_updates["current_task"]

            # 태스크 진행상황 업데이트
            if "task_progress" in context_updates:
                session_context.task_progress.update(context_updates["task_progress"])

            session_context.last_updated = datetime.now()

            # 에이전트 상태 업데이트
            if agent_id in self.agent_states:
                agent_state = self.agent_states[agent_id]
                agent_state.last_activity = datetime.now()

                if "local_context" in context_updates:
                    agent_state.local_context.update(context_updates["local_context"])

                if "current_operation" in context_updates:
                    agent_state.current_operation = context_updates["current_operation"]

            # 세션 저장
            await self._save_session(session_context)

        # 컨텍스트 업데이트 이벤트 발생
        await self._emit_event("context_update", {
            "session_id": session_id,
            "agent_id": agent_id,
            "update_keys": list(context_updates.keys())
        })

        logger.info(f"🔄 컨텍스트 업데이트: {agent_id} → {session_id}")

        return {
            "success": True,
            "session_id": session_id,
            "agent_id": agent_id,
            "updated_context": asdict(session_context),
            "timestamp": session_context.last_updated.isoformat()
        }

    async def _sync_agent_state(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 상태 동기화"""
        agent_id = params.get("agent_id")
        session_id = params.get("session_id")

        if not agent_id:
            raise ContextSyncError("Missing agent_id")

        # 특정 세션의 상태만 동기화
        if session_id:
            session_context = await self._load_session(session_id)
            if not session_context:
                raise ContextSyncError(f"Session {session_id} not found")

            agent_state = self.agent_states.get(agent_id)
            if not agent_state or agent_state.session_id != session_id:
                raise ContextSyncError(f"Agent {agent_id} not in session {session_id}")

            return {
                "success": True,
                "agent_id": agent_id,
                "session_id": session_id,
                "session_context": asdict(session_context),
                "agent_state": asdict(agent_state),
                "sync_timestamp": datetime.now().isoformat()
            }

        # 모든 세션 상태 동기화
        agent_sessions = {}
        for sid, session in self.sessions.items():
            if agent_id in session.participating_agents:
                agent_sessions[sid] = asdict(session)

        agent_state = self.agent_states.get(agent_id)

        return {
            "success": True,
            "agent_id": agent_id,
            "participating_sessions": agent_sessions,
            "agent_state": asdict(agent_state) if agent_state else None,
            "sync_timestamp": datetime.now().isoformat()
        }

    async def _broadcast_to_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """세션 내 브로드캐스트"""
        session_id = params.get("session_id")
        sender_agent = params.get("agent_id")
        message = params.get("message")
        message_type = params.get("message_type", "general")

        if not session_id or not sender_agent or not message:
            raise ContextSyncError("Missing required parameters for broadcast")

        # 세션 존재 확인
        session_context = await self._load_session(session_id)
        if not session_context:
            raise ContextSyncError(f"Session {session_id} not found")

        # 브로드캐스트 메시지 구성
        broadcast_message = {
            "id": f"msg_{datetime.now().timestamp()}",
            "session_id": session_id,
            "sender": sender_agent,
            "message": message,
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            "recipients": list(session_context.participating_agents - {sender_agent})
        }

        # 각 에이전트의 pending_messages에 추가
        delivered_count = 0
        for agent_id in session_context.participating_agents:
            if agent_id != sender_agent and agent_id in self.agent_states:
                self.agent_states[agent_id].pending_messages.append(broadcast_message)
                delivered_count += 1

        # 브로드캐스트 이벤트 발생
        await self._emit_event("message_broadcast", {
            "session_id": session_id,
            "sender": sender_agent,
            "message_id": broadcast_message["id"],
            "recipients": broadcast_message["recipients"]
        })

        logger.info(f"📢 세션 브로드캐스트: {sender_agent} → {session_id} ({delivered_count}명)")

        return {
            "success": True,
            "message_id": broadcast_message["id"],
            "session_id": session_id,
            "delivered_count": delivered_count,
            "recipients": broadcast_message["recipients"]
        }

    async def _get_session_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """세션 정보 조회"""
        session_id = params.get("session_id")

        if not session_id:
            # 모든 활성 세션 목록 반환
            active_sessions = {}
            for sid, session in self.sessions.items():
                active_sessions[sid] = {
                    "user_id": session.user_id,
                    "created_at": session.created_at.isoformat(),
                    "last_updated": session.last_updated.isoformat(),
                    "participating_agents": list(session.participating_agents),
                    "current_task": session.current_task
                }

            return {
                "success": True,
                "active_sessions": active_sessions,
                "total_sessions": len(active_sessions)
            }

        # 특정 세션 정보 반환
        session_context = await self._load_session(session_id)
        if not session_context:
            raise ContextSyncError(f"Session {session_id} not found")

        return {
            "success": True,
            "session_info": asdict(session_context),
            "agent_states": {
                agent_id: asdict(state) for agent_id, state in self.agent_states.items()
                if state.session_id == session_id
            }
        }

    async def _get_agent_states(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 상태 조회"""
        agent_id = params.get("agent_id")

        if agent_id:
            # 특정 에이전트 상태 반환
            agent_state = self.agent_states.get(agent_id)
            if not agent_state:
                return {"success": True, "agent_state": None}

            return {
                "success": True,
                "agent_state": asdict(agent_state),
                "pending_messages_count": len(agent_state.pending_messages)
            }

        # 모든 에이전트 상태 반환
        all_states = {}
        for aid, state in self.agent_states.items():
            all_states[aid] = {
                "session_id": state.session_id,
                "last_activity": state.last_activity.isoformat(),
                "current_operation": state.current_operation,
                "pending_messages_count": len(state.pending_messages)
            }

        return {
            "success": True,
            "agent_states": all_states,
            "total_agents": len(all_states)
        }

    async def _save_session(self, session_context: SessionContext):
        """세션 저장"""
        if self.use_redis and self.redis_client:
            # Redis에 저장
            key = f"session:{session_context.session_id}"
            data = pickle.dumps(session_context)
            self.redis_client.setex(key, self.sync_config["session_timeout"], data)
        else:
            # 메모리에 저장
            self.sessions[session_context.session_id] = session_context

    async def _load_session(self, session_id: str) -> Optional[SessionContext]:
        """세션 로드"""
        if self.use_redis and self.redis_client:
            # Redis에서 로드
            key = f"session:{session_id}"
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        else:
            # 메모리에서 로드
            return self.sessions.get(session_id)

    async def _remove_session(self, session_id: str):
        """세션 제거"""
        if self.use_redis and self.redis_client:
            key = f"session:{session_id}"
            self.redis_client.delete(key)
        else:
            if session_id in self.sessions:
                del self.sessions[session_id]

        # 동기화 락 제거
        if session_id in self.sync_locks:
            del self.sync_locks[session_id]

    async def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """이벤트 발생"""
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }

        await self.event_queue.put(event)

    async def _background_sync_worker(self):
        """백그라운드 동기화 작업"""
        while True:
            try:
                # 이벤트 처리
                while not self.event_queue.empty():
                    event = await self.event_queue.get()
                    event_type = event["type"]

                    if event_type in self.event_handlers:
                        handler = self.event_handlers[event_type]
                        await handler(event["data"])

                # 정기적 동기화
                await self._periodic_sync()

                await asyncio.sleep(self.sync_config["sync_interval"])

            except Exception as e:
                logger.error(f"❌ 백그라운드 동기화 오류: {e}")
                await asyncio.sleep(5)

    async def _periodic_sync(self):
        """정기적 동기화"""
        # 에이전트 활성 상태 확인
        current_time = datetime.now()
        timeout_threshold = timedelta(seconds=self.sync_config["agent_timeout"])

        inactive_agents = []
        for agent_id, agent_state in self.agent_states.items():
            if current_time - agent_state.last_activity > timeout_threshold:
                inactive_agents.append(agent_id)

        # 비활성 에이전트 정리
        for agent_id in inactive_agents:
            await self._cleanup_inactive_agent(agent_id)

    async def _cleanup_inactive_agent(self, agent_id: str):
        """비활성 에이전트 정리"""
        if agent_id in self.agent_states:
            agent_state = self.agent_states[agent_id]
            session_id = agent_state.session_id

            # 세션에서 제거
            await self._leave_session({
                "session_id": session_id,
                "agent_id": agent_id
            })

            logger.info(f"🧹 비활성 에이전트 정리: {agent_id}")

    async def _cleanup_worker(self):
        """정리 작업 워커"""
        while True:
            try:
                if self.sync_config["auto_cleanup"]:
                    await self._cleanup_expired_sessions()

                await asyncio.sleep(300)  # 5분마다 정리

            except Exception as e:
                logger.error(f"❌ 정리 작업 오류: {e}")
                await asyncio.sleep(60)

    async def _cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        current_time = datetime.now()
        timeout_threshold = timedelta(seconds=self.sync_config["session_timeout"])

        expired_sessions = []
        for session_id, session_context in self.sessions.items():
            if current_time - session_context.last_updated > timeout_threshold:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self._remove_session(session_id)
            logger.info(f"🧹 만료된 세션 정리: {session_id}")

    # 이벤트 핸들러들
    async def _handle_agent_join(self, event_data: Dict[str, Any]):
        """에이전트 참여 이벤트 처리"""
        logger.info(f"📥 Agent Join Event: {event_data}")

    async def _handle_agent_leave(self, event_data: Dict[str, Any]):
        """에이전트 떠남 이벤트 처리"""
        logger.info(f"📤 Agent Leave Event: {event_data}")

    async def _handle_context_update(self, event_data: Dict[str, Any]):
        """컨텍스트 업데이트 이벤트 처리"""
        logger.info(f"🔄 Context Update Event: {event_data}")

    async def _handle_message_broadcast(self, event_data: Dict[str, Any]):
        """메시지 브로드캐스트 이벤트 처리"""
        logger.info(f"📢 Message Broadcast Event: {event_data}")

    async def _handle_task_progress(self, event_data: Dict[str, Any]):
        """태스크 진행 이벤트 처리"""
        logger.info(f"📊 Task Progress Event: {event_data}")

# 전역 인스턴스
context_sync_tool = ContextSyncTool()

# MCP Tool 인터페이스
async def carfin_context_sync(params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    CarFin MCP Tool: 컨텍스트 동기화

    사용 예시:
    - 세션 생성: {"action": "create_session", "user_id": "user123", "agent_id": "vehicle_expert"}
    - 세션 참여: {"action": "join_session", "session_id": "session_123", "agent_id": "finance_expert"}
    - 컨텍스트 업데이트: {"action": "update_context", "session_id": "session_123", "context_data": {...}}
    - 브로드캐스트: {"action": "broadcast", "session_id": "session_123", "message": "추천 완료"}
    """
    return await context_sync_tool.execute(params, context)
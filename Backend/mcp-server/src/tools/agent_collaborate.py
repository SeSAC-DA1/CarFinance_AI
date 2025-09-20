"""
CarFin MCP Tool: Agent Collaborate
에이전트 간 실시간 협업 및 메시지 교환
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger("CarFin-MCP.AgentCollaborate")

class AgentCollaborateError(Exception):
    """에이전트 협업 관련 에러"""
    pass

class AgentCollaborateTool:
    """에이전트 간 협업 도구"""

    def __init__(self):
        self.message_queue = {}  # agent_id -> messages
        self.collaboration_sessions = {}  # session_id -> session_info
        self.agent_status = {}  # agent_id -> status
        logger.info("✅ AgentCollaborate Tool 초기화 완료")

    async def execute(self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        에이전트 협업 실행

        params:
            action: 'send_message' | 'get_messages' | 'create_session' | 'join_session'
            agent_id: 에이전트 ID
            target_agent: 대상 에이전트 (send_message용)
            message: 메시지 내용
            session_id: 협업 세션 ID
        """
        try:
            action = params.get("action")

            if action == "send_message":
                return await self._send_message(params)
            elif action == "get_messages":
                return await self._get_messages(params)
            elif action == "create_session":
                return await self._create_collaboration_session(params)
            elif action == "join_session":
                return await self._join_session(params)
            elif action == "broadcast":
                return await self._broadcast_message(params)
            else:
                raise AgentCollaborateError(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"❌ AgentCollaborate 실행 실패: {e}")
            raise AgentCollaborateError(f"Tool execution failed: {e}")

    async def _send_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 간 메시지 전송"""
        agent_id = params.get("agent_id")
        target_agent = params.get("target_agent")
        message = params.get("message")
        message_type = params.get("message_type", "general")

        if not all([agent_id, target_agent, message]):
            raise AgentCollaborateError("Missing required parameters: agent_id, target_agent, message")

        # 메시지 구조화
        structured_message = {
            "id": f"msg_{datetime.now().timestamp()}",
            "from": agent_id,
            "to": target_agent,
            "content": message,
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            "status": "sent"
        }

        # 대상 에이전트 메시지 큐에 추가
        if target_agent not in self.message_queue:
            self.message_queue[target_agent] = []

        self.message_queue[target_agent].append(structured_message)

        logger.info(f"📨 메시지 전송: {agent_id} → {target_agent}")

        return {
            "success": True,
            "message_id": structured_message["id"],
            "delivery_status": "delivered",
            "timestamp": structured_message["timestamp"]
        }

    async def _get_messages(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 메시지 수신"""
        agent_id = params.get("agent_id")
        limit = params.get("limit", 10)
        mark_read = params.get("mark_read", True)

        if not agent_id:
            raise AgentCollaborateError("Missing required parameter: agent_id")

        messages = self.message_queue.get(agent_id, [])

        # 최신 메시지부터 반환
        recent_messages = messages[-limit:] if limit > 0 else messages

        # 읽음 표시
        if mark_read:
            for msg in recent_messages:
                msg["status"] = "read"

        logger.info(f"📬 메시지 수신: {agent_id} - {len(recent_messages)}건")

        return {
            "success": True,
            "messages": recent_messages,
            "total_count": len(messages),
            "unread_count": len([m for m in messages if m.get("status") != "read"])
        }

    async def _create_collaboration_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """협업 세션 생성"""
        session_name = params.get("session_name", f"session_{datetime.now().timestamp()}")
        creator_agent = params.get("agent_id")
        participants = params.get("participants", [])

        if not creator_agent:
            raise AgentCollaborateError("Missing required parameter: agent_id")

        session_id = f"collab_{datetime.now().timestamp()}"

        session_info = {
            "id": session_id,
            "name": session_name,
            "creator": creator_agent,
            "participants": list(set([creator_agent] + participants)),
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "shared_context": {},
            "message_history": []
        }

        self.collaboration_sessions[session_id] = session_info

        logger.info(f"🤝 협업 세션 생성: {session_id} by {creator_agent}")

        return {
            "success": True,
            "session_id": session_id,
            "session_info": session_info
        }

    async def _join_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """협업 세션 참여"""
        session_id = params.get("session_id")
        agent_id = params.get("agent_id")

        if not all([session_id, agent_id]):
            raise AgentCollaborateError("Missing required parameters: session_id, agent_id")

        if session_id not in self.collaboration_sessions:
            raise AgentCollaborateError(f"Session not found: {session_id}")

        session = self.collaboration_sessions[session_id]

        if agent_id not in session["participants"]:
            session["participants"].append(agent_id)

        # 참여 메시지 기록
        join_message = {
            "type": "system",
            "content": f"{agent_id} joined the session",
            "timestamp": datetime.now().isoformat()
        }
        session["message_history"].append(join_message)

        logger.info(f"👥 세션 참여: {agent_id} → {session_id}")

        return {
            "success": True,
            "session_id": session_id,
            "participants": session["participants"],
            "shared_context": session["shared_context"]
        }

    async def _broadcast_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """전체 에이전트에게 메시지 브로드캐스트"""
        agent_id = params.get("agent_id")
        message = params.get("message")
        session_id = params.get("session_id")

        if not all([agent_id, message]):
            raise AgentCollaborateError("Missing required parameters: agent_id, message")

        delivered_count = 0

        if session_id and session_id in self.collaboration_sessions:
            # 세션 참여자들에게만 브로드캐스트
            session = self.collaboration_sessions[session_id]
            targets = [p for p in session["participants"] if p != agent_id]
        else:
            # 모든 활성 에이전트에게 브로드캐스트
            targets = [aid for aid in self.agent_status.keys() if aid != agent_id]

        # 각 대상에게 메시지 전송
        for target in targets:
            try:
                await self._send_message({
                    "agent_id": agent_id,
                    "target_agent": target,
                    "message": message,
                    "message_type": "broadcast"
                })
                delivered_count += 1
            except Exception as e:
                logger.warning(f"⚠️ 브로드캐스트 실패: {target} - {e}")

        logger.info(f"📢 브로드캐스트 완료: {agent_id} → {delivered_count}명")

        return {
            "success": True,
            "delivered_count": delivered_count,
            "total_targets": len(targets),
            "timestamp": datetime.now().isoformat()
        }

    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """에이전트 등록"""
        self.agent_status[agent_id] = {
            "status": "active",
            "info": agent_info,
            "last_seen": datetime.now().isoformat()
        }

        if agent_id not in self.message_queue:
            self.message_queue[agent_id] = []

        logger.info(f"✅ 에이전트 등록: {agent_id}")

    def unregister_agent(self, agent_id: str):
        """에이전트 등록 해제"""
        if agent_id in self.agent_status:
            self.agent_status[agent_id]["status"] = "inactive"

        logger.info(f"❌ 에이전트 등록 해제: {agent_id}")

    def get_active_agents(self) -> List[str]:
        """활성 에이전트 목록 반환"""
        return [
            aid for aid, info in self.agent_status.items()
            if info.get("status") == "active"
        ]

# 전역 인스턴스
agent_collaborate_tool = AgentCollaborateTool()

# MCP Tool 인터페이스
async def carfin_agent_collaborate(params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    CarFin MCP Tool: 에이전트 협업

    사용 예시:
    - 메시지 전송: {"action": "send_message", "agent_id": "vehicle_expert", "target_agent": "finance_expert", "message": "차량 정보"}
    - 메시지 수신: {"action": "get_messages", "agent_id": "finance_expert"}
    - 협업 세션 생성: {"action": "create_session", "agent_id": "gemini_agent", "session_name": "차량추천협업"}
    """
    return await agent_collaborate_tool.execute(params, context)
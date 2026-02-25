# agent_with_forced_skill.py
import os
import re
import yaml
from pathlib import Path
from typing import TypedDict, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI  # 可替换为 Qwen / Ollama / DashScope
from langgraph.graph import StateGraph, END


# ----------------------------
# 1. Agent 状态定义（新增 forced_skill 字段）
# ----------------------------
class AgentState(TypedDict):
    user_input: str
    forced_skill: Optional[str]          # 👈 新增：如果非 None，则跳过路由
    selected_skill: Optional[str]
    final_response: Optional[str]


# ----------------------------
# 2. 本地 Skill 管理器（不变）
# ----------------------------
class LocalSkillManager:
    def __init__(self, skills_dir: str = "~/.local/skills"):
        self.skills_dir = Path(skills_dir).expanduser()
        self.skills = self._load_skills()

    def _load_skills(self):
        skills = {}
        for folder in self.skills_dir.iterdir():
            if folder.is_dir():
                skill_file = folder / "SKILL.md"
                if skill_file.exists():
                    try:
                        text = skill_file.read_text(encoding="utf-8")
                        match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
                        if match:
                            meta = yaml.safe_load(match.group(1))
                            body = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL).strip()
                            skills[folder.name] = {
                                "name": meta.get("name", folder.name),
                                "description": meta.get("description", ""),
                                "instructions": body
                            }
                    except Exception as e:
                        print(f"⚠️ 加载技能 {folder.name} 失败: {e}")
        return skills

    def get_instructions(self, skill_name: str) -> str:
        if skill_name not in self.skills:
            raise ValueError(f"技能 '{skill_name}' 未在本地找到，请检查 ~/.local/skills/ 目录")
        return self.skills[skill_name]["instructions"]

    def list_skills(self):
        return list(self.skills.keys())


# ----------------------------
# 3. 节点函数
# ----------------------------
skill_manager = LocalSkillManager()
# 替换为你自己的模型，例如 Qwen：
# from langchain_community.chat_models import ChatTongyi
# llm = ChatTongyi(model="qwen-max", temperature=0.7)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


def decide_route_or_force(state: AgentState) -> Literal["execute_skill", "route_fallback"]:
    """
    判断是强制使用技能，还是走自动路由
    """
    if state.get("forced_skill"):
        # 强制模式：直接使用指定技能
        state["selected_skill"] = state["forced_skill"]
        return "execute_skill"
    else:
        # 自动路由模式：尝试匹配
        return "route_fallback"


def route_fallback(state: AgentState) -> Literal["execute_skill", "fallback"]:
    """自动路由逻辑（仅在未强制指定时调用）"""
    user_input = state["user_input"]
    skill_desc = "\n".join(
        f"- `{name}`: {info['description']}"
        for name, info in skill_manager.skills.items()
    )

    if not skill_desc:
        return "fallback"

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 AI Agent 的技能路由器。请根据用户请求，从以下可用技能中选择最匹配的一个。\n\n可用技能：\n{skills}\n\n如果都不匹配，回答 NONE。"),
        ("user", "用户请求：{input}\n\n请只输出技能名称（如 xiaohongshu）或 NONE：")
    ])

    chain = router_prompt | llm
    response = chain.invoke({"skills": skill_desc, "input": user_input})
    selected = response.content.strip()

    if selected in skill_manager.skills:
        state["selected_skill"] = selected
        return "execute_skill"
    else:
        return "fallback"


def execute_skill(state: AgentState):
    skill_name = state["selected_skill"]
    instructions = skill_manager.get_instructions(skill_name)
    user_input = state["user_input"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你正在使用「{skill_name}」技能。请严格遵守以下操作规范：\n\n{instructions}"),
        ("user", "{input}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "skill_name": skill_name,
        "instructions": instructions,
        "input": user_input
    })

    state["final_response"] = response.content
    return state


def fallback(state: AgentState):
    response = llm.invoke([HumanMessage(content=state["user_input"])])
    state["final_response"] = response.content
    return state


# ----------------------------
# 4. 构建 Graph
# ----------------------------
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("decide_route_or_force", lambda s: s)
    workflow.add_node("route_fallback", route_fallback)
    workflow.add_node("execute_skill", execute_skill)
    workflow.add_node("fallback", fallback)

    workflow.set_entry_point("decide_route_or_force")

    workflow.add_conditional_edges(
        "decide_route_or_force",
        decide_route_or_force,
        {
            "execute_skill": "execute_skill",
            "route_fallback": "route_fallback"
        }
    )

    workflow.add_conditional_edges(
        "route_fallback",
        lambda s: s,  # 直接返回字符串分支
        {
            "execute_skill": "execute_skill",
            "fallback": "fallback"
        }
    )

    workflow.add_edge("execute_skill", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


# ----------------------------
# 5. 使用示例
# ----------------------------
if __name__ == "__main__":
    app = build_graph()

    # ✅ 方式 1：自动路由
    print("【自动路由】")
    result1 = app.invoke({
        "user_input": "帮我写一篇关于护手霜的小红书爆款文案",
        "forced_skill": None,
        "selected_skill": None,
        "final_response": None
    })
    print(result1["final_response"])

    print("\n" + "="*60 + "\n")

    # ✅ 方式 2：强制使用某个 Skill（即使描述不匹配）
    print("【强制使用 xiaohongshu 技能】")
    result2 = app.invoke({
        "user_input": "总结一下量子力学的基本原理",  # 这显然不是小红书场景
        "forced_skill": "xiaohongshu",               # 但强制用它！
        "selected_skill": None,
        "final_response": None
    })
    print(result2["final_response"])

from deepagents import create_deep_agent


from langchain_core.tools import tool
from typing import List, Optional
import json

# Simple in-memory todo storage for demonstration
# In production, Deep Agents use persistent storage
TODO_STORE = {}

@tool
def write_todos(todos: List[dict]) -> str:
    """Create a list of todos for tracking task progress.
    
    Args:
        todos: List of todo items, each with 'title' and optional 'description'
    
    Returns:
        Confirmation message with todo IDs
    """
    created = []
    for i, todo in enumerate(todos):
        todo_id = f"todo_{len(TODO_STORE) + i + 1}"
        TODO_STORE[todo_id] = {
            "id": todo_id,
            "title": todo.get("title", "Untitled"),
            "description": todo.get("description", ""),
            "status": "pending"
        }
        created.append(todo_id)
    return f"Created {len(created)} todos: {', '.join(created)}"

@tool
def update_todo(todo_id: str, status: Literal["pending", "in_progress", "completed"]) -> str:
    """Update the status of a todo item.
    
    Args:
        todo_id: The ID of the todo to update
        status: New status (pending, in_progress, completed)
    
    Returns:
        Confirmation message
    """
    if todo_id not in TODO_STORE:
        return f"Todo {todo_id} not found"
    TODO_STORE[todo_id]["status"] = status
    return f"Updated {todo_id} to {status}"

@tool
def list_todos() -> str:
    """List all todos with their current status.
    
    Returns:
        Formatted list of all todos
    """
    if not TODO_STORE:
        return "No todos found"
    
    result = []
    for todo_id, todo in TODO_STORE.items():
        status_emoji = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo["status"], "❓")
        result.append(f"{emoji} [{todo_id}] {todo['title']} ({todo['status']})")
    return "\n".join(result)


#---------Planning---------

# Create some wellness todos
result = write_todos.invoke({
    "todos": [
        {"title": "Assess current sleep patterns", "description": "Review user's sleep schedule and quality"},
        {"title": "Research sleep improvement strategies", "description": "Find evidence-based techniques"},
        {"title": "Create personalized sleep plan", "description": "Combine findings into actionable steps"},
    ]
})

# Simulate progress
update_todo.invoke({"todo_id": "todo_1", "status": "completed"})
update_todo.invoke({"todo_id": "todo_2", "status": "in_progress"})

print("After updates:")
print(list_todos.invoke({}))

#---------Context Management---------

import os
from pathlib import Path

# Create a workspace directory for our agent
WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)

@tool
def ls(path: str = ".") -> str:
    """List contents of a directory.
    
    Args:
        path: Directory path to list (default: current directory)
    
    Returns:
        List of files and directories
    """
    target = WORKSPACE / path
    if not target.exists():
        return f"Directory not found: {path}"
    
    items = []
    for item in sorted(target.iterdir()):
        prefix = "[DIR]" if item.is_dir() else "[FILE]"
        size = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
        items.append(f"{prefix} {item.name}{size}")
    
    return "\n".join(items) if items else "(empty directory)"

@tool
def read_file(path: str) -> str:
    """Read contents of a file.
    
    Args:
        path: Path to the file to read
    
    Returns:
        File contents
    """
    target = WORKSPACE / path
    if not target.exists():
        return f"File not found: {path}"
    return target.read_text()

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file (creates or overwrites).
    
    Args:
        path: Path to the file to write
        content: Content to write to the file
    
    Returns:
        Confirmation message
    """
    target = WORKSPACE / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} characters to {path}"

@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Edit a file by replacing text.
    
    Args:
        path: Path to the file to edit
        old_text: Text to find and replace
        new_text: Replacement text
    
    Returns:
        Confirmation message
    """
    target = WORKSPACE / path
    if not target.exists():
        return f"File not found: {path}"
    
    content = target.read_text()
    if old_text not in content:
        return f"Text not found in {path}"
    
    new_content = content.replace(old_text, new_text, 1)
    target.write_text(new_content)
    return f"Updated {path}"



# Create a research notes file
notes = """# Sleep Research Notes

## Key Findings
- Adults need 7-9 hours of sleep
- Consistent sleep schedule is important
- Blue light affects melatonin production

## TODO
- [ ] Review individual user needs
- [ ] Create personalized recommendations
"""

result = write_file.invoke({"path": "research/sleep_notes.md", "content": notes})


#---------Basic Deep Agent---------

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model

# Configure the filesystem backend to use our workspace directory
# IMPORTANT: virtual_mode=True is required to actually restrict paths to root_dir
# Without it, agents can still write anywhere on the filesystem!
workspace_path = Path("workspace").absolute()
filesystem_backend = FilesystemBackend(
    root_dir=str(workspace_path),
    virtual_mode=True  # This is required to sandbox file operations!
)

# Combine our custom tools (for todo tracking)
# Note: Deep Agents has built-in file tools (ls, read_file, write_file, edit_file)
# that will use the configured FilesystemBackend
custom_tools = [
    write_todos,
    update_todo,
    list_todos,
]

# Create a basic Deep Agent
wellness_agent = create_deep_agent(
    model=init_chat_model("anthropic:claude-sonnet-4-20250514"),
    tools=custom_tools,
    backend=filesystem_backend,  # Configure where files are stored
    system_prompt="""You are a Personal Wellness Assistant that helps users improve their health.

When given a complex task:
1. First, create a todo list to track your progress
2. Work through each task, updating status as you go
3. Save important findings to files for reference
4. Provide a clear summary when complete

Be thorough but concise. Always explain your reasoning."""
)


# Reset todo store for fresh demo
TODO_STORE.clear()

# Test with a multi-step wellness task
result = wellness_agent.invoke({
    "messages": [{
        "role": "user",
        "content": """I want to improve my sleep quality. I currently:
- Go to bed at inconsistent times (10pm-1am)
- Use my phone in bed
- Often feel tired in the morning

Please create a personalized sleep improvement plan for me and save it to a file."""
    }]
})


#---------Subagent Spawning---------

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model

# Define specialized subagent configurations
# Note: Subagents inherit the backend from the parent agent
research_subagent = {
    "name": "research-agent",
    "description": "Use this agent to research wellness topics in depth. It can read documents and synthesize information.",
    "system_prompt": """You are a wellness research specialist. Your job is to:
    1. Find relevant information in provided documents
    2. Synthesize findings into clear summaries
    3. Cite sources when possible

    Be thorough but concise. Focus on evidence-based information.""",
    "tools": [],  # Uses built-in file tools from backend
    "model": "openai:gpt-4o-mini",  # Cheaper model for research
}

writing_subagent = {
    "name": "writing-agent",
    "description": "Use this agent to create well-structured documents, plans, and guides.",
    "system_prompt": """You are a wellness content writer. Your job is to:
    1. Take research findings and turn them into clear, actionable content
    2. Structure information for easy understanding
    3. Use formatting (headers, bullets, etc.) effectively

    Write in a supportive, encouraging tone.""",
    "tools": [],  # Uses built-in file tools from backend
    "model": "anthropic:claude-sonnet-4-20250514",
}

# Create a coordinator agent that can spawn subagents
coordinator_agent = create_deep_agent(
    model=init_chat_model("anthropic:claude-sonnet-4-20250514"),
    tools=[write_todos, update_todo, list_todos],
    backend=filesystem_backend,  # Use the same backend - subagents inherit it
    subagents=[research_subagent, writing_subagent],
    system_prompt="""You are a Wellness Project Coordinator. Your role is to:
    1. Break down complex wellness requests into subtasks
    2. Delegate research to the research-agent
    3. Delegate content creation to the writing-agent
    4. Coordinate the overall workflow using todos

    Use subagents for specialized work rather than doing everything yourself.
    This keeps the work organized and the results high-quality."""
)

# Reset for demo
TODO_STORE.clear()

# Test the coordinator with a complex task
result = coordinator_agent.invoke({
    "messages": [{
        "role": "user",
        "content": """Create a comprehensive morning routine guide for better energy.
        
    The guide should:
    1. Research the science behind morning routines
    2. Include practical steps for exercise, nutrition, and mindset
    3. Be saved as a well-formatted markdown file"""
    }]
})
print("Coordinator response:")
print(result["messages"][-1].content)

# Check the results
print("Final todo status:")
print(list_todos.invoke({}))

print("\nGenerated files in workspace:")
for f in sorted(WORKSPACE.iterdir()):
    if f.is_file():
        print(f"  [FILE] {f.name} ({f.stat().st_size} bytes)")
    elif f.is_dir():
        print(f"  [DIR] {f.name}/")




#---------Long Term Memory---------


from langgraph.store.memory import InMemoryStore

# Create a memory store
memory_store = InMemoryStore()

# Store user profile
user_id = "user_alex"
profile_namespace = (user_id, "profile")

memory_store.put(profile_namespace, "name", {"value": "Alex"})
memory_store.put(profile_namespace, "goals", {
    "primary": "improve energy levels",
    "secondary": "better sleep"
})
memory_store.put(profile_namespace, "conditions", {
    "dietary": ["vegetarian"],
    "medical": ["mild anxiety"]
})
memory_store.put(profile_namespace, "preferences", {
    "exercise_time": "morning",
    "communication_style": "detailed"
})

# Retrieve and display
for item in memory_store.search(profile_namespace):
    print(f"  {item.key}: {item.value}")


# Create memory-aware tools
from langgraph.store.base import BaseStore

@tool
def get_user_profile(user_id: str) -> str:
    """Retrieve a user's wellness profile from long-term memory.
    
    Args:
        user_id: The user's unique identifier
    
    Returns:
        User profile as formatted text
    """
    namespace = (user_id, "profile")
    items = list(memory_store.search(namespace))
    
    if not items:
        return f"No profile found for {user_id}"
    
    result = [f"Profile for {user_id}:"]
    for item in items:
        result.append(f"  {item.key}: {item.value}")
    return "\n".join(result)

@tool
def save_user_preference(user_id: str, key: str, value: str) -> str:
    """Save a user preference to long-term memory.
    
    Args:
        user_id: The user's unique identifier
        key: The preference key
        value: The preference value
    
    Returns:
        Confirmation message
    """
    namespace = (user_id, "preferences")
    memory_store.put(namespace, key, {"value": value})
    return f"Saved preference '{key}' for {user_id}"

# Create a memory-enhanced agent
memory_tools = [
    get_user_profile,
    save_user_preference,
    write_todos,
    update_todo,
    list_todos,
]

memory_agent = create_deep_agent(
    model=init_chat_model("anthropic:claude-sonnet-4-20250514"),
    tools=memory_tools,
    backend=filesystem_backend,  # Use workspace for file operations
    system_prompt="""You are a Personal Wellness Assistant with long-term memory.

    At the start of each conversation:
    1. Check the user's profile to understand their goals and conditions
    2. Personalize all advice based on their profile
    3. Save any new preferences they mention

    Always reference stored information to show you remember the user."""
)

# Test the memory agent
TODO_STORE.clear()

result = memory_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Hi! My user_id is user_alex. What exercise routine would you recommend for me?"
    }]
})

print("Agent response:")
print(result["messages"][-1].content)



#---------On Demand Capabilities---------

# Let's look at the skills we created
skills_dir = Path("skills")


for skill_dir in skills_dir.iterdir():
    if skill_dir.is_dir():
        skill_file = skill_dir / "SKILL.md"
        if skill_file.exists():
            content = skill_file.read_text()
            # Extract name and description from frontmatter
            lines = content.split("\n")
            name = ""
            desc = ""
            for line in lines:
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip()
                if line.startswith("description:"):
                    desc = line.split(":", 1)[1].strip()
            print(f"  - {name}: {desc}")

# Read the wellness-assessment skill
skill_content = Path("skills/wellness-assessment/SKILL.md").read_text()



# Create a skill-aware tool
@tool
def load_skill(skill_name: str) -> str:
    """Load a skill's instructions for a specialized task.
    
    Available skills:
    - wellness-assessment: Assess user wellness and create recommendations
    - meal-planning: Create personalized meal plans
    
    Args:
        skill_name: Name of the skill to load
    
    Returns:
        Skill instructions
    """
    skill_path = Path(f"skills/{skill_name}/SKILL.md")
    if not skill_path.exists():
        available = [d.name for d in Path("skills").iterdir() if d.is_dir()]
        return f"Skill '{skill_name}' not found. Available: {', '.join(available)}"
    
    return skill_path.read_text()




# Create an agent that can load and use skills
skill_agent = create_deep_agent(
    model=init_chat_model("anthropic:claude-sonnet-4-20250514"),
    tools=[
        load_skill,
        write_todos,
        update_todo,
        list_todos,
    ],
    backend=filesystem_backend,  # Use workspace for file operations
    system_prompt="""You are a wellness assistant with access to specialized skills.

    When a user asks for something that matches a skill:
    1. Load the appropriate skill using load_skill()
    2. Follow the skill's instructions carefully
    3. Save outputs as specified in the skill

    Available skills:
    - wellness-assessment: For comprehensive wellness evaluations
    - meal-planning: For creating personalized meal plans

    If no skill matches, use your general wellness knowledge."""
)


# Test with a skill-appropriate request
TODO_STORE.clear()

result = skill_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "I'd like a wellness assessment. I'm a 35-year-old office worker who sits most of the day, has trouble sleeping, and wants to lose 15 pounds. I'm vegetarian and have no major health conditions."
    }]
})

print("Agent response:")
print(result["messages"][-1].content)


#---------Using Deepagents-cli---------


# Check if CLI is installed
import subprocess

try:
    result = subprocess.run(["deepagents", "--version"], capture_output=True, text=True)
    print(f"deepagents-cli version: {result.stdout.strip()}")
except FileNotFoundError:
    print("deepagents-cli not installed. Install with:")
    print("  uv pip install deepagents-cli")
    print("  # or")
    print("  pip install deepagents-cli")


    #-------- Building a complete Deep Agent System---------

    # Define specialized wellness subagents
    # Subagents inherit the backend from the parent, so they use the same workspace
    exercise_specialist = {
        "name": "exercise-specialist",
        "description": "Expert in exercise science, workout programming, and physical fitness. Use for exercise-related questions and plan creation.",
        "system_prompt": """You are an exercise specialist with expertise in:
    - Workout programming for different fitness levels
    - Exercise form and safety
    - Progressive overload principles
    - Recovery and injury prevention

    Always consider the user's fitness level and any physical limitations.
    Provide clear, actionable exercise instructions.""",
        "tools": [],  # Uses built-in file tools from backend
        "model": "openai:gpt-4o-mini",
    }

    nutrition_specialist = {
        "name": "nutrition-specialist",
        "description": "Expert in nutrition science, meal planning, and dietary optimization. Use for food-related questions and meal plans.",
        "system_prompt": """You are a nutrition specialist with expertise in:
    - Macro and micronutrient balance
    - Meal planning and preparation
    - Dietary restrictions and alternatives
    - Nutrition timing for performance

    Always respect dietary restrictions and preferences.
    Focus on practical, achievable meal suggestions.""",
        "tools": [],  # Uses built-in file tools from backend
        "model": "openai:gpt-4o-mini",
    }

    mindfulness_specialist = {
        "name": "mindfulness-specialist",
        "description": "Expert in stress management, sleep optimization, and mental wellness. Use for stress, sleep, and mental health questions.",
        "system_prompt": """You are a mindfulness and mental wellness specialist with expertise in:
    - Stress reduction techniques
    - Sleep hygiene and optimization
    - Meditation and breathing exercises
    - Work-life balance strategies

    Be supportive and non-judgmental.
    Provide practical techniques that can be implemented immediately.""",
        "tools": [],  # Uses built-in file tools from backend
        "model": "openai:gpt-4o-mini",
    }

# Create the Wellness Coach coordinator
wellness_coach = create_deep_agent(
    model=init_chat_model("anthropic:claude-sonnet-4-20250514"),
    tools=[
        # Planning
        write_todos,
        update_todo,
        list_todos,
        # Long-term Memory
        get_user_profile,
        save_user_preference,
        # Skills
        load_skill,
    ],
    backend=filesystem_backend,  # All file ops go to workspace
    subagents=[exercise_specialist, nutrition_specialist, mindfulness_specialist],
    system_prompt="""You are a Personal Wellness Coach that coordinates comprehensive wellness programs.

    ## Your Role
    - Understand each user's unique goals, constraints, and preferences
    - Create personalized, multi-week wellness programs
    - Coordinate between exercise, nutrition, and mindfulness specialists
    - Track progress and adapt recommendations

    ## Workflow
    1. **Initial Assessment**: Get user profile and understand their situation
    2. **Planning**: Create a todo list for the program components
    3. **Delegation**: Use specialists for domain-specific content:
    - exercise-specialist: Workout plans and fitness guidance
    - nutrition-specialist: Meal plans and dietary advice
    - mindfulness-specialist: Stress and sleep optimization
    4. **Integration**: Combine specialist outputs into a cohesive program
    5. **Documentation**: Save all plans and recommendations to files

    ## Important
    - Always check user profile first for context
    - Respect any medical conditions or dietary restrictions
    - Provide clear, actionable recommendations
    - Save progress to files so users can reference later"""
)

# Test the complete system
TODO_STORE.clear()

result = wellness_coach.invoke({
    "messages": [{
        "role": "user",
        "content": """Hi! My user_id is user_alex. I'd like you to create a 2-week wellness program for me.

    I want to focus on:
    1. Building a consistent exercise routine (I can exercise 3x per week for 30 mins)
    2. Improving my diet (remember I'm vegetarian)
    3. Better managing my work stress and improving my sleep

    Please create comprehensive plans for each area and save them as separate files I can reference."""
        }]
    })

print("Wellness Coach response:")
print(result["messages"][-1].content)


# Review what was created
print("=" * 60)
print("FINAL TODO STATUS")
print("=" * 60)
print(list_todos.invoke({}))

print("\n" + "=" * 60)
print("GENERATED FILES")
print("=" * 60)
for f in sorted(WORKSPACE.iterdir()):
    if f.is_file():
        print(f"  [FILE] {f.name} ({f.stat().st_size} bytes)")
    elif f.is_dir():
        print(f"  [DIR] {f.name}/")

# Read one of the generated files
files = list(WORKSPACE.glob("*.md"))
if files:
    print(f"\nContents of {files[0].name}:")
    print("=" * 60)
    print(files[0].read_text()[:2000] + "..." if len(files[0].read_text()) > 2000 else files[0].read_text())
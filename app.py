from flask import Flask, request, jsonify
from flask_cors import CORS
from crewai import Agent, Crew, Process, Task
from langchain_groq import ChatGroq
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) 

os.environ["GROQ_API_KEY"] = "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"

if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
    raise ValueError("GROQ_API_KEY environment variable is not set or is empty")

llm = ChatGroq(temperature=0.7, model_name="llama3-8b-8192")

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    service = data.get('service')
    task_description = data.get('taskDescription')
    age_group = data.get('ageGroup')

    if service == "Therapy Sessions":
        role = "Therapist"
        goal = f"""Provide professional counseling sessions to help individuals in the {age_group} age group manage mental health challenges. 
        Offer personalized support, guidance, and evidence-based strategies tailored to their needs. 
        Incorporate mental exercises such as mindfulness meditation, cognitive behavioral therapy (CBT) exercises, gratitude journaling, art therapy, memory exercises, progressive muscle relaxation (PMR), social engagement activities, and gentle physical exercise."""
    elif service == "Mindfulness Programs":
        role = "Mindfulness Coach"
        goal = f"Teach techniques and practices that promote mental well-being through mindfulness to the {age_group} age group. Facilitate sessions to enhance self-awareness, reduce stress, and improve overall emotional balance."
    elif service == "Support Groups":
        role = "Support Group Facilitator"
        goal = f"Create a safe and supportive environment for individuals in the {age_group} age group facing similar mental health challenges. Foster connection, empathy, and mutual support among group members through facilitated discussions and activities."
    else:
        return "Invalid service", 400

    agent_instance = Agent(
        role=role,
        goal=goal,
        backstory=f"""As a {role}, your role is to {goal.lower()}. You are dedicated to supporting individuals in the {age_group} age group in their mental health journey, providing professional guidance, fostering resilience, and promoting holistic well-being.""",
        verbose=False,  # Set verbose to False to reduce logging
        allow_delegation=False,
        llm=llm,
        max_iter=5,
        memory=True,
    )

    task = Task(
        description=task_description,
        expected_output="Output based on the task description.",
        input_value=task_description,
        input_type="str",
        agent=agent_instance,
    )

    crew = Crew(
        agents=[agent_instance],
        tasks=[task],
        verbose=0,  # Set verbose to 0 to reduce logging
        process=Process.sequential,
        full_output=False,  # Set full_output to False to avoid excessive output
        share_crew=False,
        manager_llm=llm,
        max_iter=10,
    )

    results = crew.kickoff()

    return results

if __name__ == '__main__':
    app.run(debug=True, port=5000)

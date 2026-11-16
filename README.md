# SmartDesk 

A Smartdesk AI Agent to resolve IT/HR Employee Queries and also a tool to create Jira ticket if not in scope.

TechStack:
React
FastApi
Langraph
Rag
MCP
Langsmith


Backend (terminal 1):
    cd backend
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Frontend (terminal 2):
    cd frontend
    npm install
    npm run dev

Open http://localhost:5173 and click Send: the reply is "Hello World".

## System Architecture

![System Architecture Diagram](./images/architecture.png)

## Langraph 
[./images/langraph.png]

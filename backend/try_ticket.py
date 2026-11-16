import jira_mcp

for tool, args in [
    ("jira_get_all_projects", {}),
    ("jira_search", {"jql": "project is not EMPTY ORDER BY created DESC", "limit": 5}),
]:
    try:
        print(f"--- {tool}")
        print(jira_mcp.call_tool(tool, args)[:1500])
    except Exception as e:
        print("failed:", e)
"""FastMCP server exposing GitHub issue tools.

Every tool returns a dict (never a bare list) so the tool message sent back to the
LLM is never empty, which some providers (e.g. Groq) reject.
"""

from typing import Any, Optional

from fastmcp import FastMCP

from app.db import record_audit
from app.github import GitHubClient

mcp = FastMCP("github-issues")
github_client = GitHubClient()


@mcp.tool(
    name="create_issue",
    description="Create a new GitHub issue. Uses the default repository if repo is omitted.",
)
async def create_issue(
    title: str,
    body: str,
    repo: Optional[str] = None,
    labels: Optional[list[str]] = None,
    assignees: Optional[list[str]] = None,
) -> dict[str, Any]:
    res = await github_client.create_issue(repo, title, body, labels, assignees)
    await record_audit(
        action="create",
        repo=res["repo"],
        issue_number=res.get("issue_number"),
        payload={"title": title, "body": body, "labels": labels, "url": res.get("url")},
    )
    return res


@mcp.tool(
    name="get_issue",
    description="Fetch one issue and its comments. Uses the default repository if repo is omitted.",
)
async def get_issue(issue_number: int, repo: Optional[str] = None) -> dict[str, Any]:
    return await github_client.get_issue(repo, issue_number)


@mcp.tool(
    name="list_issues",
    description=(
        "List issues in a repository. Returns the repo, a count and the issues. "
        "Uses the default repository if repo is omitted. Labels is a comma-separated string."
    ),
)
async def list_issues(
    repo: Optional[str] = None,
    state: str = "open",
    labels: Optional[str] = None,
    limit: int = 30,
) -> dict[str, Any]:
    label_list = [x.strip() for x in labels.split(",") if x.strip()] if labels else None
    issues = await github_client.list_issues(repo, state=state, labels=label_list, limit=limit)
    return {
        "repo": github_client.resolve_repo(repo),
        "state": state,
        "count": len(issues),
        "issues": issues,
    }


@mcp.tool(
    name="count_issues",
    description=(
        "Get the exact number of issues (pull requests excluded) in a repository. "
        "state is 'open', 'closed' or 'all'. Uses the default repository if repo is omitted."
    ),
)
async def count_issues(repo: Optional[str] = None, state: str = "open") -> dict[str, Any]:
    return await github_client.count_issues(repo, state)


@mcp.tool(
    name="search_issues",
    description="Search issues in a repository, e.g. to check for duplicates before creating one.",
)
async def search_issues(query: str, repo: Optional[str] = None) -> dict[str, Any]:
    results = await github_client.search_issues(repo, query)
    return {"repo": github_client.resolve_repo(repo), "count": len(results), "results": results}


@mcp.tool(
    name="comment_issue",
    description="Add a comment to an existing issue. Uses the default repository if repo is omitted.",
)
async def comment_issue(issue_number: int, body: str, repo: Optional[str] = None) -> dict[str, Any]:
    res = await github_client.comment_issue(repo, issue_number, body)
    await record_audit(
        action="comment",
        repo=res["repo"],
        issue_number=issue_number,
        payload={"body": body, "url": res.get("url")},
    )
    return res


@mcp.tool(
    name="close_issue",
    description=(
        "Close an existing issue with a reason ('completed' or 'not_planned') and an optional "
        "comment. Uses the default repository if repo is omitted."
    ),
)
async def close_issue(
    issue_number: int,
    repo: Optional[str] = None,
    reason: Optional[str] = "completed",
    comment: Optional[str] = None,
) -> dict[str, Any]:
    res = await github_client.close_issue(repo, issue_number, reason=reason, comment=comment)
    await record_audit(
        action="close",
        repo=res["repo"],
        issue_number=issue_number,
        payload={"reason": reason, "comment": comment},
    )
    return res


if __name__ == "__main__":
    mcp.run()

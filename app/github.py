"""Async GitHub REST API client."""

from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import get_settings


class GitHubError(Exception):
    def __init__(self, message: str, status_code: int = 500, response_data: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class RepoNotAllowedError(GitHubError):
    def __init__(self, repo: str):
        super().__init__(
            f"Repository '{repo}' is not in the allowed repositories list.", status_code=403
        )


class IssueNotFoundError(GitHubError):
    def __init__(self, repo: str, issue_number: int):
        super().__init__(
            f"Issue #{issue_number} not found in repository '{repo}'.", status_code=404
        )


class GitHubClient:
    def __init__(self, token: Optional[str] = None):
        self.settings = get_settings()
        self.token = token or self.settings.github_token
        self.base_url = "https://api.github.com"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "IssueHandler-Agent",
        }

    def resolve_repo(self, repo: Optional[str]) -> str:
        """Use the given repo, fall back to DEFAULT_REPO, and validate it."""
        resolved = (repo or self.settings.default_repo or "").strip()
        if not resolved:
            raise GitHubError(
                "No repository given and DEFAULT_REPO is not set. "
                "Ask the user for the repository in 'owner/repo' format.",
                status_code=400,
            )
        if resolved.count("/") != 1:
            raise GitHubError(
                f"'{resolved}' is not a valid repository. Use the 'owner/repo' format.",
                status_code=400,
            )
        if not self.settings.is_repo_allowed(resolved):
            raise RepoNotAllowedError(resolved)
        return resolved

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict[str, Any]] = None,
        json_data: Optional[Any] = None,
    ) -> Any:
        url = f"{self.base_url}{path}" if path.startswith("/") else f"{self.base_url}/{path}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=self._headers(),
                params=params,
                json=json_data,
            )

            # Retry on rate limiting
            if response.status_code in (403, 429) and "rate limit" in response.text.lower():
                response.raise_for_status()

            if response.status_code == 404:
                raise GitHubError(
                    f"Not found: {method} {path}. Check that the repository/issue exists "
                    "and that the GitHub token has access to it.",
                    status_code=404,
                )
            if response.is_error:
                try:
                    msg = response.json().get("message", response.text)
                except Exception:
                    msg = response.text
                raise GitHubError(
                    f"GitHub API error [{response.status_code}]: {msg}",
                    status_code=response.status_code,
                    response_data=msg,
                )

            if response.status_code == 204:
                return {}
            return response.json()

    async def create_issue(
        self,
        repo: Optional[str],
        title: str,
        body: str,
        labels: Optional[list[str]] = None,
        assignees: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        repo = self.resolve_repo(repo)
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees
        res = await self._request("POST", f"/repos/{repo}/issues", json_data=payload)
        return {
            "repo": repo,
            "issue_number": res.get("number"),
            "title": res.get("title"),
            "url": res.get("html_url"),
            "state": res.get("state"),
            "created_at": res.get("created_at"),
        }

    async def get_issue(self, repo: Optional[str], issue_number: int) -> dict[str, Any]:
        repo = self.resolve_repo(repo)
        try:
            issue = await self._request("GET", f"/repos/{repo}/issues/{issue_number}")
        except GitHubError as e:
            if e.status_code == 404:
                raise IssueNotFoundError(repo, issue_number)
            raise

        comments: list[dict[str, Any]] = []
        if issue.get("comments", 0) > 0:
            raw_comments = await self._request(
                "GET", f"/repos/{repo}/issues/{issue_number}/comments"
            )
            comments = [
                {
                    "id": c.get("id"),
                    "user": (c.get("user") or {}).get("login"),
                    "body": c.get("body"),
                    "created_at": c.get("created_at"),
                }
                for c in raw_comments
            ]

        return {
            "repo": repo,
            "number": issue.get("number"),
            "title": issue.get("title"),
            "body": issue.get("body") or "",
            "state": issue.get("state"),
            "state_reason": issue.get("state_reason"),
            "url": issue.get("html_url"),
            "user": (issue.get("user") or {}).get("login"),
            "labels": [
                lb.get("name") if isinstance(lb, dict) else lb for lb in issue.get("labels", [])
            ],
            "comments": comments,
            "created_at": issue.get("created_at"),
            "updated_at": issue.get("updated_at"),
        }

    async def list_issues(
        self,
        repo: Optional[str],
        state: str = "open",
        labels: Optional[list[str] | str] = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        repo = self.resolve_repo(repo)
        params: dict[str, Any] = {"state": state, "per_page": min(limit, 100)}
        if labels:
            params["labels"] = ",".join(labels) if isinstance(labels, list) else labels

        raw_issues = await self._request("GET", f"/repos/{repo}/issues", params=params)
        results: list[dict[str, Any]] = []
        for issue in raw_issues:
            # GitHub's issues endpoint also returns pull requests; skip them
            if "pull_request" in issue:
                continue
            results.append(
                {
                    "number": issue.get("number"),
                    "title": issue.get("title"),
                    "body": (issue.get("body") or "")[:200],
                    "state": issue.get("state"),
                    "url": issue.get("html_url"),
                    "labels": [
                        lb.get("name") if isinstance(lb, dict) else lb
                        for lb in issue.get("labels", [])
                    ],
                }
            )
            if len(results) >= limit:
                break
        return results

    async def count_issues(self, repo: Optional[str], state: str = "open") -> dict[str, Any]:
        """Exact issue count (pull requests excluded) using the search API."""
        repo = self.resolve_repo(repo)
        q = f"repo:{repo} is:issue"
        if state in ("open", "closed"):
            q += f" state:{state}"
        res = await self._request("GET", "/search/issues", params={"q": q, "per_page": 1})
        return {"repo": repo, "state": state, "count": res.get("total_count", 0)}

    async def search_issues(self, repo: Optional[str], query: str) -> list[dict[str, Any]]:
        repo = self.resolve_repo(repo)
        q = f"repo:{repo} is:issue {query}"
        res = await self._request("GET", "/search/issues", params={"q": q, "per_page": 5})
        return [
            {
                "number": item.get("number"),
                "title": item.get("title"),
                "state": item.get("state"),
                "url": item.get("html_url"),
            }
            for item in res.get("items", [])
        ]

    async def comment_issue(
        self, repo: Optional[str], issue_number: int, body: str
    ) -> dict[str, Any]:
        repo = self.resolve_repo(repo)
        res = await self._request(
            "POST", f"/repos/{repo}/issues/{issue_number}/comments", json_data={"body": body}
        )
        return {
            "repo": repo,
            "comment_id": res.get("id"),
            "url": res.get("html_url"),
            "created_at": res.get("created_at"),
        }

    async def close_issue(
        self,
        repo: Optional[str],
        issue_number: int,
        reason: Optional[str] = "completed",
        comment: Optional[str] = None,
    ) -> dict[str, Any]:
        repo = self.resolve_repo(repo)
        if comment:
            await self.comment_issue(repo, issue_number, comment)

        payload = {"state": "closed", "state_reason": reason or "completed"}
        res = await self._request(
            "PATCH", f"/repos/{repo}/issues/{issue_number}", json_data=payload
        )
        return {
            "repo": repo,
            "issue_number": res.get("number"),
            "title": res.get("title"),
            "state": res.get("state"),
            "state_reason": res.get("state_reason"),
            "url": res.get("html_url"),
        }
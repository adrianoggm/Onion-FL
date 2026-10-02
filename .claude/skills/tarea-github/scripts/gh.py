"""GitHub helper for the task/#N workflow, over the REST API with the git credential.

No `gh` CLI needed: the token comes from Git Credential Manager (`git credential fill`).
Run from the repository root:  python .claude/skills/tarea-github/scripts/gh.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo() -> str:
    url = _git("remote", "get-url", "origin")
    m = re.search(r"github\.com[:/](?P<repo>[^/]+/[^/]+?)(?:\.git)?$", url)
    if not m:
        raise SystemExit(f"origin is not a GitHub remote: {url}")
    return m["repo"]


def _token(owner: str) -> str:
    env = {**os.environ, "GCM_INTERACTIVE": "never", "GIT_TERMINAL_PROMPT": "0"}
    # bytes input: a BOM (PowerShell pipes) makes git reject the protocol field
    for request in (
        b"protocol=https\nhost=github.com\n\n",
        f"protocol=https\nhost=github.com\nusername={owner}\n\n".encode(),
    ):
        out = subprocess.run(
            ["git", "credential", "fill"], input=request, capture_output=True, env=env
        ).stdout.decode()
        for line in out.splitlines():
            if line.startswith("password="):
                return line.split("=", 1)[1]
    raise SystemExit(
        "No github.com credential in the git credential helper (log in once with a git push over HTTPS)."
    )


REPO = _repo()
TOKEN = _token(REPO.split("/")[0])


def api(method: str, path: str, body: dict | None = None):
    url = path if path.startswith("https://") else f"https://api.github.com{path}"
    req = urllib.request.Request(
        url,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read() or b"null"), resp.headers
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"{method} {path} -> {exc.code}: {exc.read().decode()[:400]}"
        ) from exc


def paged(path: str) -> list:
    items, page = [], 1
    while True:
        sep = "&" if "?" in path else "?"
        batch, _ = api("GET", f"{path}{sep}per_page=100&page={page}")
        items += batch
        if len(batch) < 100:
            return items
        page += 1


R = f"/repos/{REPO}"
DEPS_RE = re.compile(r"\*\*Depende de:\*\*\s*(.+)")


def cmd_whoami(_a) -> None:
    user, headers = api("GET", "/user")
    repo, _ = api("GET", R)
    perms = repo.get("permissions", {})
    print(
        f"user={user['login']} scopes={headers.get('X-OAuth-Scopes')} repo={REPO} push={perms.get('push')} admin={perms.get('admin')}"
    )


def cmd_issues(a) -> None:
    query = f"{R}/issues?state={a.state}"
    if a.milestone:
        ms = {m["title"]: m["number"] for m in paged(f"{R}/milestones?state=all")}
        query += f"&milestone={ms[a.milestone]}"
    for it in paged(query):
        if "pull_request" not in it:
            labels = ",".join(lab["name"] for lab in it["labels"])
            print(f"#{it['number']:<4} {it['state']:<6} {it['title']}  [{labels}]")


def cmd_issue(a) -> None:
    it, _ = api("GET", f"{R}/issues/{a.number}")
    ms = it["milestone"]["title"] if it["milestone"] else "—"
    print(
        f"#{it['number']} [{it['state']}] {it['title']}\nlabels: {', '.join(lab['name'] for lab in it['labels'])} | milestone: {ms}\n"
    )
    print(it["body"] or "")
    m = DEPS_RE.search(it["body"] or "")
    if m:
        deps = re.findall(r"#(\d+)", m[1])
        if deps:
            print("\nDependency status:")
            for d in deps:
                dep, _ = api("GET", f"{R}/issues/{d}")
                print(f"  #{d} {dep['state']}: {dep['title']}")


def parse_backlog(text: str) -> list[dict]:
    issues = []
    for block in re.split(r"^## \[", text, flags=re.M)[1:]:
        head, _, rest = block.partition("\n")
        key, _, title = head.partition("] ")
        meta = re.search(
            r"^Meta: labels=(?P<l>[^·]+) · milestone=(?P<m>\S+) · depende=(?P<d>.+)$\n",
            rest,
            flags=re.M,
        )
        if not meta:
            raise SystemExit(
                f"[{key}] has no 'Meta: labels=… · milestone=… · depende=…' line"
            )
        body = re.split(r"\n---\n|\n## ", rest[meta.end() :])[0].strip()
        deps = (
            []
            if meta["d"].strip() in {"—", "-"}
            else [d.strip() for d in meta["d"].split(",")]
        )
        issues.append(
            {
                "key": key,
                "title": f"[{key}] {title.strip()}",
                "labels": [x.strip() for x in meta["l"].split(",")],
                "milestone": meta["m"],
                "deps": deps,
                "body": body,
            }
        )
    keys = [i["key"] for i in issues]
    for i in issues:
        for d in i["deps"]:
            if d.lstrip("#").isdigit():
                continue
            if d not in keys or keys.index(d) > keys.index(i["key"]):
                raise SystemExit(
                    f"[{i['key']}] depends on {d}, which is not defined earlier in the file"
                )
    return issues


def cmd_backlog(a) -> None:
    issues = parse_backlog(Path(a.file).read_text(encoding="utf-8"))
    labels = {lab["name"] for lab in paged(f"{R}/labels")}
    ms = {m["title"]: m["number"] for m in paged(f"{R}/milestones?state=all")}
    existing = {
        it["title"]: it["number"]
        for it in paged(f"{R}/issues?state=all")
        if "pull_request" not in it
    }
    missing_labels = sorted({lab for i in issues for lab in i["labels"]} - labels)
    missing_ms = sorted({i["milestone"] for i in issues} - set(ms))
    print(
        f"{len(issues)} issues | new labels: {missing_labels or '—'} | new milestones: {missing_ms or '—'}"
    )
    if a.dry_run:
        for i in issues:
            print(
                f"  {'skip (exists)' if i['title'] in existing else 'create'}: {i['title']}  deps={i['deps'] or '—'}"
            )
        return
    for lab in missing_labels:
        api("POST", f"{R}/labels", {"name": lab, "color": "ededed"})
    for title in missing_ms:
        ms[title] = api("POST", f"{R}/milestones", {"title": title})[0]["number"]
    numbers: dict[str, int] = {}
    for i in issues:
        if i["title"] in existing:
            numbers[i["key"]] = existing[i["title"]]
            continue
        deps = (
            ", ".join(d if d.startswith("#") else f"#{numbers[d]}" for d in i["deps"])
            or "—"
        )
        body = f"{i['body']}\n\n---\n**Depende de:** {deps}\n\nRama `task/#<número de este issue>` desde `develop`; PR hacia `develop`."
        created, _ = api(
            "POST",
            f"{R}/issues",
            {
                "title": i["title"],
                "body": body,
                "labels": i["labels"],
                "milestone": ms[i["milestone"]],
            },
        )
        numbers[i["key"]] = created["number"]
        print(f"  + #{created['number']} {i['title']}")
        time.sleep(1.5)  # GitHub secondary rate limit on content creation


def _open_pr(number: int) -> dict | None:
    owner = REPO.split("/")[0]
    pulls = paged(f"{R}/pulls?state=open&head={owner}:task/%23{number}")
    return pulls[0] if pulls else None


def cmd_pr(a) -> None:
    """Open (or refresh) the PR of task/#N with the issue's labels, milestone and link."""
    branch = f"task/#{a.number}"
    issue, _ = api("GET", f"{R}/issues/{a.number}")
    _git("fetch", "origin", a.base)
    subjects = _git("log", f"origin/{a.base}..{branch}", "--reverse", "--format=%s")
    if not subjects:
        raise SystemExit(f"{branch} has no commits ahead of origin/{a.base}")
    repo, _ = api("GET", R)
    # Closing keywords only link/close issues when the PR targets the default branch
    keyword = "Closes" if a.base == repo["default_branch"] else "Refs"
    body = f"{keyword} #{a.number} — {issue['title']}\n\n{subjects}"
    if a.note:
        body += f"\n\n**Verificación local:** {a.note}"
    pr = _open_pr(a.number)
    if pr:
        pr, _ = api("PATCH", f"{R}/pulls/{pr['number']}", {"body": body})
    else:
        payload = {
            "title": branch,
            "head": branch,
            "base": a.base,
            "body": body,
            "draft": a.draft,
        }
        pr, _ = api("POST", f"{R}/pulls", payload)
    user, _ = api("GET", "/user")
    labels = [lab["name"] for lab in issue["labels"]]
    milestone = issue["milestone"]["number"] if issue["milestone"] else None
    meta = {"labels": labels, "milestone": milestone, "assignees": [user["login"]]}
    api("PATCH", f"{R}/issues/{pr['number']}", meta)
    ms = issue["milestone"]["title"] if issue["milestone"] else "—"
    print(f"PR #{pr['number']} -> {pr['base']['ref']}: {pr['html_url']}")
    print(
        f"  labels={labels} milestone={ms} assignee={user['login']} body: {keyword} #{a.number}"
    )


def cmd_pr_status(a) -> None:
    pulls = paged(f"{R}/pulls?state=all&head={REPO.split('/')[0]}:task/%23{a.number}")
    if not pulls:
        raise SystemExit(f"no PR for task/#{a.number}")
    pr, _ = api("GET", f"{R}/pulls/{pulls[0]['number']}")
    print(
        f"PR #{pr['number']} {pr['state']} merged={pr['merged']} mergeable={pr.get('mergeable_state')} -> {pr['base']['ref']}\n{pr['html_url']}"
    )
    runs, _ = api("GET", f"{R}/commits/{pr['head']['sha']}/check-runs")
    for run in runs["check_runs"]:
        print(f"  {run['name']}: {run['status']} {run['conclusion'] or ''}")


def cmd_merge(a) -> None:
    pr, _ = api("GET", f"{R}/pulls/{a.pr}")
    runs, _ = api("GET", f"{R}/commits/{pr['head']['sha']}/check-runs")
    bad = [
        f"{r['name']}={r['conclusion'] or r['status']}"
        for r in runs["check_runs"]
        if r["conclusion"] not in {"success", "skipped", "neutral"}
    ]
    # CI only runs on PRs into main; task PRs into develop are gated by local verification
    no_ci_expected = pr["base"]["ref"] != "main" and not runs["check_runs"]
    if no_ci_expected:
        print(
            f"PR #{a.pr} -> {pr['base']['ref']}: no CI by design; local verification is the gate"
        )
    elif (bad or not runs["check_runs"]) and not a.force:
        raise SystemExit(
            f"PR #{a.pr} checks not green: {bad or 'no checks yet'} (use --force to override)"
        )
    res, _ = api("PUT", f"{R}/pulls/{a.pr}/merge", {"merge_method": "merge"})
    print(res["message"])


def cmd_close(a) -> None:
    if a.pr:
        pr, _ = api("GET", f"{R}/pulls/{a.pr}")
        if not pr["merged"]:
            raise SystemExit(f"PR #{a.pr} is not merged; not closing #{a.number}")
    note = f"Integrado en `develop` vía #{a.pr}." if a.pr else "Cerrado."
    api("POST", f"{R}/issues/{a.number}/comments", {"body": note})
    api(
        "PATCH",
        f"{R}/issues/{a.number}",
        {"state": "closed", "state_reason": "completed"},
    )
    print(f"#{a.number} closed")
    if a.pr:
        try:
            api("DELETE", f"{R}/git/refs/heads/task/%23{a.number}")
            print(f"remote branch task/#{a.number} deleted")
        except SystemExit as exc:  # already deleted
            print(f"remote branch not deleted: {exc}")


def _milestone_notes(title: str) -> tuple[int, str]:
    ms = {m["title"]: m["number"] for m in paged(f"{R}/milestones?state=all")}
    closed = [
        it
        for it in paged(f"{R}/issues?state=closed&milestone={ms[title]}")
        if "pull_request" not in it
    ]
    notes = "\n".join(
        f"- #{it['number']} {it['title']}"
        for it in sorted(closed, key=lambda x: x["number"])
    )
    return ms[title], notes or "- (no closed issues)"


def _release_body(a, issues: str) -> str:
    """Hand-written notes (``--notes FILE``) first, then the milestone's issues."""
    if not getattr(a, "notes", None):
        return issues
    text = Path(a.notes).read_text(encoding="utf-8").strip()
    return f"{text}\n\n## Issues\n\n{issues}"


def cmd_release_pr(a) -> None:
    _, notes = _milestone_notes(a.version)
    notes = _release_body(a, notes)
    pr, _ = api(
        "POST",
        f"{R}/pulls",
        {
            "title": f"release {a.version}",
            "head": "develop",
            "base": "main",
            "body": f"Release {a.version}\n\n{notes}",
        },
    )
    print(f"PR #{pr['number']}: {pr['html_url']}")


def cmd_release(a) -> None:
    number, notes = _milestone_notes(a.version)
    notes = _release_body(a, notes)
    rel, _ = api(
        "POST",
        f"{R}/releases",
        {
            "tag_name": a.version,
            "target_commitish": "main",
            "name": a.version,
            "body": notes,
        },
    )
    api("PATCH", f"{R}/milestones/{number}", {"state": "closed"})
    print(f"release {a.version}: {rel['html_url']} (milestone closed)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami", help="credential user, scopes and repo permissions")
    s = sub.add_parser("issues", help="list issues")
    s.add_argument("--milestone")
    s.add_argument("--state", default="open", choices=["open", "closed", "all"])
    s = sub.add_parser("issue", help="show an issue and the state of its dependencies")
    s.add_argument("number", type=int)
    s = sub.add_parser(
        "backlog",
        help="create labels, milestones and issues from a backlog .md (idempotent by title)",
    )
    s.add_argument("file")
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser(
        "pr",
        help="open or refresh the PR of task/#N: issue labels, milestone, assignee, issue link + commit subjects",
    )
    s.add_argument("number", type=int)
    s.add_argument("--base", default="develop")
    s.add_argument("--draft", action="store_true")
    s.add_argument("--note", help="local verification result, appended to the body")
    s = sub.add_parser("pr-status", help="PR state and CI check runs for task/#N")
    s.add_argument("number", type=int)
    s = sub.add_parser(
        "merge",
        help="merge a PR with a merge commit (never squash); PRs into main need green CI",
    )
    s.add_argument("pr", type=int)
    s.add_argument(
        "--force", action="store_true", help="merge even if checks are not green"
    )
    s = sub.add_parser(
        "close",
        help="close an issue after its PR is merged into develop (with --pr: also deletes the remote task branch)",
    )
    s.add_argument("number", type=int)
    s.add_argument("--pr", type=int)
    s = sub.add_parser(
        "release-pr", help="open the develop -> main PR for a milestone version"
    )
    s.add_argument("version")
    s.add_argument(
        "--notes", help="Markdown release notes to put before the issue list"
    )
    s = sub.add_parser(
        "release",
        help="after the release PR is merged: tag main, publish notes, close the milestone",
    )
    s.add_argument("version")
    s.add_argument(
        "--notes", help="Markdown release notes to put before the issue list"
    )
    a = p.parse_args()
    {
        "whoami": cmd_whoami,
        "issues": cmd_issues,
        "issue": cmd_issue,
        "backlog": cmd_backlog,
        "pr": cmd_pr,
        "pr-status": cmd_pr_status,
        "merge": cmd_merge,
        "close": cmd_close,
        "release-pr": cmd_release_pr,
        "release": cmd_release,
    }[a.cmd](a)


if __name__ == "__main__":
    main()

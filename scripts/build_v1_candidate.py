"""Build the deterministic V1.0 candidate dataset without fabricating secondary labels."""

# ruff: noqa: RUF001 -- Chinese benchmark text intentionally uses full-width punctuation.

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_route_reliability.hashing import tool_corpus_sha256  # noqa: E402
from mcp_route_reliability.models import QueryCase, ToolRecord  # noqa: E402

COMMIT = "d73f99efbfd40c3aa1b61e88728b3d49fb52608f"
LICENSE = "Apache-2.0 OR MIT"
ACCESS_DATE = "2026-09-22"
SEED = 20260921


def object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def string(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


def string_array(description: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": description}


def tool(
    tool_id: str,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    server: str,
    domain: str,
    source_path: str,
) -> ToolRecord:
    return ToolRecord(
        tool_id=tool_id,
        name=name,
        description=description,
        input_schema=input_schema,
        server=server,
        domain=domain,
        source_url=(f"https://github.com/modelcontextprotocol/servers/blob/{COMMIT}/{source_path}"),
        source_commit=COMMIT,
        license=LICENSE,
    )


RELATION = {
    "type": "object",
    "properties": {
        "from": string("Entity where the relation starts"),
        "to": string("Entity where the relation ends"),
        "relationType": string("Active-voice relation type"),
    },
    "required": ["from", "to", "relationType"],
    "additionalProperties": False,
}


NEW_TOOLS = [
    tool(
        "filesystem.read_media_file",
        "Read Media File",
        "Read one image, audio, or other binary file as a base64 MCP content block.",
        object_schema({"path": string("Media file path")}, ["path"]),
        "modelcontextprotocol/filesystem",
        "filesystem",
        "src/filesystem/index.ts",
    ),
    tool(
        "filesystem.create_directory",
        "Create Directory",
        "Create a directory and any missing parent directories, succeeding if it already exists.",
        object_schema({"path": string("Directory path")}, ["path"]),
        "modelcontextprotocol/filesystem",
        "filesystem",
        "src/filesystem/index.ts",
    ),
    tool(
        "filesystem.list_directory_with_sizes",
        "List Directory with Sizes",
        "List immediate directory entries with file sizes, sorted by name or size.",
        object_schema(
            {
                "path": string("Directory path"),
                "sortBy": {
                    "type": "string",
                    "enum": ["name", "size"],
                    "default": "name",
                    "description": "Entry sort order",
                },
            },
            ["path"],
        ),
        "modelcontextprotocol/filesystem",
        "filesystem",
        "src/filesystem/index.ts",
    ),
    tool(
        "filesystem.directory_tree",
        "Directory Tree",
        "Return a recursive JSON tree of files and directories with optional exclusions.",
        object_schema(
            {
                "path": string("Root directory path"),
                "excludePatterns": string_array("Glob patterns to exclude"),
            },
            ["path"],
        ),
        "modelcontextprotocol/filesystem",
        "filesystem",
        "src/filesystem/index.ts",
    ),
    tool(
        "filesystem.move_file",
        "Move File",
        "Move or rename a file or directory without overwriting an existing destination.",
        object_schema(
            {"source": string("Source path"), "destination": string("Destination path")},
            ["source", "destination"],
        ),
        "modelcontextprotocol/filesystem",
        "filesystem",
        "src/filesystem/index.ts",
    ),
    tool(
        "filesystem.get_file_info",
        "Get File Info",
        "Return file or directory metadata such as type, size, timestamps, and permissions.",
        object_schema({"path": string("File or directory path")}, ["path"]),
        "modelcontextprotocol/filesystem",
        "filesystem",
        "src/filesystem/index.ts",
    ),
    tool(
        "filesystem.list_allowed_directories",
        "List Allowed Directories",
        "List the directories that the filesystem server is permitted to access.",
        object_schema({}, []),
        "modelcontextprotocol/filesystem",
        "filesystem",
        "src/filesystem/index.ts",
    ),
    tool(
        "git.diff",
        "Git Diff",
        "Show differences between branches, commits, or another Git target.",
        object_schema(
            {
                "repo_path": string("Repository path"),
                "target": string("Branch, commit, or diff target"),
                "context_lines": {"type": "integer", "default": 3},
            },
            ["repo_path", "target"],
        ),
        "modelcontextprotocol/git",
        "git",
        "src/git/src/mcp_server_git/server.py",
    ),
    tool(
        "git.commit",
        "Git Commit",
        "Record staged changes in a local Git repository with a commit message.",
        object_schema(
            {"repo_path": string("Repository path"), "message": string("Commit message")},
            ["repo_path", "message"],
        ),
        "modelcontextprotocol/git",
        "git",
        "src/git/src/mcp_server_git/server.py",
    ),
    tool(
        "git.add",
        "Git Add",
        "Add selected file contents to the Git staging area.",
        object_schema(
            {"repo_path": string("Repository path"), "files": string_array("Paths to stage")},
            ["repo_path", "files"],
        ),
        "modelcontextprotocol/git",
        "git",
        "src/git/src/mcp_server_git/server.py",
    ),
    tool(
        "git.reset",
        "Git Reset",
        "Unstage all currently staged changes without discarding working-tree edits.",
        object_schema({"repo_path": string("Repository path")}, ["repo_path"]),
        "modelcontextprotocol/git",
        "git",
        "src/git/src/mcp_server_git/server.py",
    ),
    tool(
        "git.create_branch",
        "Create Git Branch",
        "Create a new Git branch from the current or an optional base branch.",
        object_schema(
            {
                "repo_path": string("Repository path"),
                "branch_name": string("New branch name"),
                "base_branch": string("Optional base branch"),
            },
            ["repo_path", "branch_name"],
        ),
        "modelcontextprotocol/git",
        "git",
        "src/git/src/mcp_server_git/server.py",
    ),
    tool(
        "git.checkout",
        "Git Checkout",
        "Switch the local repository to an existing branch.",
        object_schema(
            {
                "repo_path": string("Repository path"),
                "branch_name": string("Existing branch name"),
            },
            ["repo_path", "branch_name"],
        ),
        "modelcontextprotocol/git",
        "git",
        "src/git/src/mcp_server_git/server.py",
    ),
    tool(
        "git.branch",
        "List Git Branches",
        "List local, remote, or all branches with optional commit containment filters.",
        object_schema(
            {
                "repo_path": string("Repository path"),
                "branch_type": {
                    "type": "string",
                    "enum": ["local", "remote", "all"],
                    "description": "Which branches to list",
                },
                "contains": string("Optional commit that branches must contain"),
                "not_contains": string("Optional commit that branches must not contain"),
            },
            ["repo_path", "branch_type"],
        ),
        "modelcontextprotocol/git",
        "git",
        "src/git/src/mcp_server_git/server.py",
    ),
    tool(
        "memory.delete_entities",
        "Delete Entities",
        "Delete named knowledge-graph entities and all relations attached to them.",
        object_schema({"entityNames": string_array("Entity names to delete")}, ["entityNames"]),
        "modelcontextprotocol/memory",
        "memory",
        "src/memory/index.ts",
    ),
    tool(
        "memory.delete_observations",
        "Delete Observations",
        "Delete selected textual observations from existing knowledge-graph entities.",
        object_schema(
            {
                "deletions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entityName": string("Entity name"),
                            "observations": string_array("Observation texts to delete"),
                        },
                        "required": ["entityName", "observations"],
                        "additionalProperties": False,
                    },
                }
            },
            ["deletions"],
        ),
        "modelcontextprotocol/memory",
        "memory",
        "src/memory/index.ts",
    ),
    tool(
        "memory.delete_relations",
        "Delete Relations",
        "Delete exact relations while preserving their endpoint entities.",
        object_schema({"relations": {"type": "array", "items": RELATION}}, ["relations"]),
        "modelcontextprotocol/memory",
        "memory",
        "src/memory/index.ts",
    ),
    tool(
        "memory.read_graph",
        "Read Graph",
        "Read every entity and relation in the knowledge graph.",
        object_schema({}, []),
        "modelcontextprotocol/memory",
        "memory",
        "src/memory/index.ts",
    ),
    tool(
        "utility.echo",
        "Echo",
        "Echo one input string unchanged for protocol testing.",
        object_schema({"message": string("Message to echo")}, ["message"]),
        "modelcontextprotocol/everything",
        "utility",
        "src/everything/tools/echo.ts",
    ),
    tool(
        "utility.get_sum",
        "Get Sum",
        "Return the numeric sum of two provided numbers.",
        object_schema(
            {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            ["a", "b"],
        ),
        "modelcontextprotocol/everything",
        "utility",
        "src/everything/tools/get-sum.ts",
    ),
]


@dataclass(frozen=True)
class Spec:
    case_type: str
    relevant: tuple[str, ...]
    en: str
    zh: str
    rationale: str
    conflict: str | None = None
    abstain_reason: str | None = None


CLEAR_INTENTS = [
    (
        "filesystem.read_text_file",
        "Read the first 20 lines of docs/guide.md",
        "读取 docs/guide.md 的前 20 行",
    ),
    (
        "filesystem.read_multiple_files",
        "Read README.md and LICENSE together in one call",
        "一次读取 README.md 和 LICENSE 两个文件",
    ),
    (
        "filesystem.write_file",
        "Overwrite build/version.txt with 1.4.0",
        "把 build/version.txt 完整覆盖为 1.4.0",
    ),
    (
        "filesystem.edit_file",
        "Replace DEBUG=true with DEBUG=false in app.env and show the diff",
        "把 app.env 中的 DEBUG=true 精确替换为 DEBUG=false 并显示差异",
    ),
    (
        "filesystem.list_directory",
        "List only the immediate children of src",
        "只列出 src 目录的直接子项",
    ),
    (
        "filesystem.search_files",
        "Recursively find every Python file under src",
        "递归查找 src 下所有 Python 文件",
    ),
    (
        "filesystem.read_media_file",
        "Read assets/logo.png as an image content block",
        "把 assets/logo.png 读取为图片内容块",
    ),
    (
        "filesystem.create_directory",
        "Create the nested directory reports/2026/september",
        "创建嵌套目录 reports/2026/september",
    ),
    (
        "filesystem.list_directory_with_sizes",
        "List Downloads entries sorted by file size",
        "按文件大小列出 Downloads 的直接子项",
    ),
    (
        "filesystem.directory_tree",
        "Return a recursive JSON tree of src excluding __pycache__",
        "返回 src 的递归 JSON 目录树并排除 __pycache__",
    ),
    ("filesystem.move_file", "Rename draft.md to final.md", "把 draft.md 重命名为 final.md"),
    (
        "filesystem.get_file_info",
        "Show size and modification time for archive.zip without reading it",
        "不读取 archive.zip 内容，只查看大小和修改时间",
    ),
    (
        "filesystem.list_allowed_directories",
        "Show every directory this filesystem server may access",
        "显示该文件系统服务器允许访问的所有目录",
    ),
    ("git.status", "Show repository working-tree and staging status", "显示仓库工作区和暂存区状态"),
    ("git.diff_unstaged", "Show only changes that have not been staged", "只显示尚未暂存的修改"),
    (
        "git.diff_staged",
        "Show the patch currently staged for commit",
        "显示当前已暂存、准备提交的补丁",
    ),
    ("git.diff", "Compare release/1.0 with main", "比较 release/1.0 与 main 的差异"),
    (
        "git.commit",
        "Commit the staged changes with message fix parser",
        "使用消息 fix parser 提交已暂存修改",
    ),
    (
        "git.add",
        "Stage src/parser.py and tests/test_parser.py",
        "暂存 src/parser.py 和 tests/test_parser.py",
    ),
    (
        "git.reset",
        "Unstage every staged change but keep local edits",
        "取消暂存所有修改但保留本地编辑",
    ),
    ("git.log", "Show the ten most recent commits", "显示最近十次提交"),
    ("git.create_branch", "Create feature/router from main", "从 main 创建 feature/router 分支"),
    ("git.checkout", "Switch to the existing branch hotfix/login", "切换到已有分支 hotfix/login"),
    ("git.show", "Show metadata and patch for commit a1b2c3d", "显示提交 a1b2c3d 的元数据和补丁"),
    ("git.branch", "List all remote Git branches", "列出所有远程 Git 分支"),
    (
        "memory.create_entities",
        "Create Alice and DataPlatform as graph entities",
        "在知识图谱中创建 Alice 和 DataPlatform 两个实体",
    ),
    (
        "memory.create_relations",
        "Create an Alice owns DataPlatform relation",
        "创建 Alice owns DataPlatform 关系",
    ),
    (
        "memory.add_observations",
        "Append the observation works remotely to Alice",
        "给 Alice 追加观察 works remotely",
    ),
    (
        "memory.delete_entities",
        "Delete the obsolete entity LegacyService and its relations",
        "删除过时实体 LegacyService 及其关联关系",
    ),
    (
        "memory.delete_observations",
        "Remove the observation works remotely from Alice",
        "从 Alice 删除观察 works remotely",
    ),
    (
        "memory.delete_relations",
        "Delete only the Alice owns DataPlatform relation",
        "只删除 Alice owns DataPlatform 关系并保留实体",
    ),
    ("memory.read_graph", "Read the complete knowledge graph", "读取完整知识图谱"),
    (
        "memory.search_nodes",
        "Search graph names, types, and observations for platform",
        "在图谱名称、类型和观察中搜索 platform",
    ),
    (
        "memory.open_nodes",
        "Open the exact graph nodes Alice and Bob",
        "按精确名称打开图谱节点 Alice 和 Bob",
    ),
    (
        "time.get_current_time",
        "Get the current time in Asia/Shanghai",
        "获取 Asia/Shanghai 当前时间",
    ),
    (
        "time.convert_time",
        "Convert 09:30 from Europe/London to Asia/Shanghai",
        "把 Europe/London 的 09:30 转换到 Asia/Shanghai",
    ),
    (
        "web.fetch",
        "Fetch https://example.com/docs as simplified Markdown",
        "抓取 https://example.com/docs 并简化为 Markdown",
    ),
    (
        "reasoning.sequential_thinking",
        "Work through the migration plan in numbered revisable steps",
        "用可修订的编号步骤推演迁移计划",
    ),
    ("utility.echo", "Echo the exact string health-check-42", "原样回显字符串 health-check-42"),
    ("utility.get_sum", "Return the sum of 19.5 and 22.5", "计算 19.5 与 22.5 的和"),
]

HARD_INTENTS = [
    (
        "filesystem.list_directory",
        "List direct children only; do not recurse or glob",
        "只列直接子项，不要递归或使用 glob",
        "list versus recursive search",
    ),
    (
        "filesystem.search_files",
        "Find matching paths recursively, not just one directory level",
        "递归查找匹配路径，不是只看一层目录",
        "recursive search versus listing",
    ),
    (
        "filesystem.read_text_file",
        "Read one text file, not a batch of paths",
        "读取单个文本文件，不要批量读取",
        "single versus multiple read",
    ),
    (
        "filesystem.read_multiple_files",
        "Read three text files together, not one at a time",
        "一次读取三个文本文件，不要逐个处理",
        "batch versus single read",
    ),
    (
        "filesystem.write_file",
        "Replace the entire file content, not selected lines",
        "替换整个文件内容，不是编辑部分行",
        "overwrite versus exact edit",
    ),
    (
        "filesystem.edit_file",
        "Replace one exact paragraph and preserve all other text",
        "精确替换一个段落并保留其余内容",
        "exact edit versus overwrite",
    ),
    (
        "filesystem.list_directory_with_sizes",
        "List immediate entries and include byte sizes",
        "列直接子项并包含文件大小",
        "sized versus plain listing",
    ),
    (
        "filesystem.directory_tree",
        "Return the full nested directory hierarchy as JSON",
        "以 JSON 返回完整嵌套目录层级",
        "tree versus one-level listing",
    ),
    (
        "filesystem.move_file",
        "Rename the existing file without rewriting its contents",
        "不重写内容，只重命名已有文件",
        "move versus write",
    ),
    (
        "filesystem.get_file_info",
        "Inspect metadata without opening file contents",
        "不打开文件内容，只检查元数据",
        "metadata versus content read",
    ),
    (
        "git.diff_staged",
        "Inspect staged changes, not working-tree-only edits",
        "检查已暂存修改，不看仅工作区修改",
        "staged versus unstaged diff",
    ),
    (
        "git.diff_unstaged",
        "Inspect working-tree edits that are not staged",
        "检查尚未暂存的工作区修改",
        "unstaged versus staged diff",
    ),
    (
        "git.diff",
        "Compare two named branches rather than show one commit",
        "比较两个命名分支，不是查看单个提交",
        "target diff versus commit show",
    ),
    (
        "git.create_branch",
        "Create a new branch but do not switch to it",
        "创建新分支但不要切换过去",
        "create versus checkout",
    ),
    (
        "git.checkout",
        "Switch to an existing branch; do not create one",
        "切换到已有分支，不要新建分支",
        "checkout versus create",
    ),
    (
        "git.add",
        "Stage files without creating a commit",
        "只暂存文件，不创建提交",
        "stage versus commit",
    ),
    (
        "git.reset",
        "Unstage changes without discarding the edits",
        "取消暂存但不要丢弃编辑",
        "reset versus status or diff",
    ),
    (
        "memory.delete_relations",
        "Delete one relation but preserve both endpoint entities",
        "删除一条关系但保留两端实体",
        "relation versus entity deletion",
    ),
    (
        "memory.delete_observations",
        "Delete one fact but preserve its entity and relations",
        "删除一条观察但保留实体和关系",
        "observation versus entity deletion",
    ),
    (
        "memory.read_graph",
        "Return every graph node and relation without a search term",
        "无需搜索词，返回全部图谱节点和关系",
        "full graph versus search",
    ),
    (
        "time.convert_time",
        "Convert a specified clock time; do not ask for now",
        "转换指定钟点，不要查询当前时间",
        "conversion versus current time",
    ),
    (
        "utility.get_sum",
        "Add two numbers rather than echo their text",
        "对两个数字求和，不要原样回显",
        "numeric sum versus echo",
    ),
]

MULTI_INTENTS = [
    (
        ("filesystem.read_text_file", "filesystem.read_multiple_files"),
        "Read notes.txt; either a single-file or batch reader can return it",
        "读取 notes.txt；单文件或批量读取工具都能完成",
    ),
    (
        ("filesystem.list_directory", "filesystem.list_directory_with_sizes"),
        "List immediate entries in tmp; extra size metadata is acceptable",
        "列出 tmp 的直接子项；附带大小信息也可以",
    ),
    (
        ("memory.search_nodes", "memory.open_nodes"),
        "Retrieve the uniquely named graph node Alice",
        "获取名称唯一的图谱节点 Alice",
    ),
    (
        ("memory.read_graph", "memory.search_nodes"),
        "Find every graph record mentioning DataPlatform; reading then filtering is acceptable",
        "找出所有提到 DataPlatform 的图谱记录；读取全图后筛选也可以",
    ),
    (
        ("git.diff", "git.show"),
        "Show the patch introduced by commit a1b2c3d",
        "显示提交 a1b2c3d 引入的补丁",
    ),
]

CLARIFY_INTENTS = [
    (
        ("filesystem.read_text_file", "filesystem.read_multiple_files"),
        "Read the file, but no path or number of files is specified",
        "读取文件，但没有给出路径或文件数量",
    ),
    (
        ("filesystem.list_directory", "filesystem.search_files"),
        "Find something in the project, but no path or matching rule is provided",
        "在项目里找东西，但没有路径或匹配规则",
    ),
    (
        ("filesystem.write_file", "filesystem.edit_file"),
        "Change the configuration, but the desired content and edit mode are unspecified",
        "修改配置，但没有说明目标内容或编辑方式",
    ),
    (
        ("git.diff_staged", "git.diff_unstaged", "git.diff"),
        "Show a Git diff, but the staged state or comparison target is missing",
        "显示 Git 差异，但没有说明暂存状态或比较目标",
    ),
    (
        ("git.create_branch",),
        "Create a branch, but the branch name is missing",
        "创建分支，但没有提供分支名",
    ),
    (
        ("git.checkout",),
        "Switch branches, but the destination branch is missing",
        "切换分支，但没有提供目标分支",
    ),
    (
        ("git.commit",),
        "Commit the staged work, but no commit message is provided",
        "提交已暂存内容，但没有提供提交信息",
    ),
    (
        ("memory.create_entities", "memory.create_relations", "memory.add_observations"),
        "Update the knowledge graph, but the intended operation is unclear",
        "更新知识图谱，但没有说明具体操作",
    ),
    (
        ("memory.delete_entities", "memory.delete_observations", "memory.delete_relations"),
        "Delete graph data, but the object to remove is ambiguous",
        "删除图谱数据，但没有说明要删除实体、观察还是关系",
    ),
    (
        ("time.get_current_time", "time.convert_time"),
        "Handle a time request, but no timezone or clock time is given",
        "处理时间请求，但没有时区或钟点",
    ),
    (("web.fetch",), "Fetch a page, but no URL is supplied", "抓取网页，但没有提供 URL"),
    (
        ("reasoning.sequential_thinking",),
        "Reason through the problem, but the problem itself is missing",
        "逐步推理，但没有给出要解决的问题",
    ),
    (
        ("utility.echo",),
        "Echo a message, but the message is missing",
        "回显消息，但没有提供消息内容",
    ),
    (
        ("utility.get_sum",),
        "Add two numbers, but one operand is missing",
        "计算两个数的和，但缺少一个操作数",
    ),
]

NO_MATCH_INTENTS = [
    ("Send an email to the finance team", "给财务团队发送邮件"),
    ("Create a calendar event for tomorrow", "创建明天的日历事件"),
    ("Forecast hourly weather for next Monday", "预测下周一逐小时天气"),
    ("Run a PostgreSQL query against the customer database", "对客户 PostgreSQL 数据库执行查询"),
    ("Deploy the service to a cloud cluster", "把服务部署到云集群"),
    ("Generate a product illustration from a text prompt", "根据文字提示生成产品插画"),
    ("Transcribe the attached meeting audio", "转录附件中的会议音频"),
    ("Charge the customer credit card", "向客户信用卡扣款"),
    ("Update a contact in the CRM", "更新 CRM 中的联系人"),
    ("Calculate a driving route on a map", "计算地图驾车路线"),
    ("Post a message to the team chat", "向团队聊天群发送消息"),
    ("Click the checkout button in a browser", "在浏览器中点击结算按钮"),
    ("Retrieve the live stock price for NVDA", "获取 NVDA 的实时股票价格"),
    ("OCR every page of a scanned PDF", "对扫描 PDF 的每一页执行 OCR"),
    ("Edit formulas in an Excel workbook", "编辑 Excel 工作簿中的公式"),
    ("Start a Docker container from an image", "从镜像启动 Docker 容器"),
    ("Scale a Kubernetes deployment to five replicas", "把 Kubernetes 部署扩容到五个副本"),
    ("Install a package from the system package manager", "使用系统包管理器安装软件包"),
]


def make_specs() -> list[Spec]:
    specs: list[Spec] = []
    for index in range(75):
        target, en, zh = CLEAR_INTENTS[index % len(CLEAR_INTENTS)]
        specs.append(
            Spec(
                "clear",
                (target,),
                en,
                zh,
                f"The complete request uniquely identifies {target}.",
            )
        )
    for index in range(35):
        target, en, zh, conflict = HARD_INTENTS[index % len(HARD_INTENTS)]
        specs.append(
            Spec(
                "hard_negative",
                (target,),
                en,
                zh,
                f"The qualifier selects {target} over a neighboring capability.",
                conflict,
            )
        )
    for index in range(20):
        targets, en, zh = MULTI_INTENTS[index % len(MULTI_INTENTS)]
        specs.append(
            Spec(
                "multi_valid",
                targets,
                en,
                zh,
                "Each listed tool can independently satisfy the complete request.",
                "Multiple tools are valid by design.",
            )
        )
    for index in range(35):
        targets, en, zh = CLARIFY_INTENTS[index % len(CLARIFY_INTENTS)]
        reason = "insufficient_information" if len(targets) == 1 else "ambiguous"
        specs.append(
            Spec(
                "clarify",
                targets,
                en,
                zh,
                (
                    "A relevant capability exists, but the request is not actionable "
                    "without clarification."
                ),
                "Required arguments or intent remain unresolved.",
                reason,
            )
        )
    for index in range(35):
        en, zh = NO_MATCH_INTENTS[index % len(NO_MATCH_INTENTS)]
        specs.append(
            Spec(
                "no_match",
                (),
                en,
                zh,
                "No tool in the frozen corpus provides this external capability.",
                abstain_reason="no_match",
            )
        )
    assert Counter(spec.case_type for spec in specs) == {
        "clear": 75,
        "hard_negative": 35,
        "multi_valid": 20,
        "clarify": 35,
        "no_match": 35,
    }
    return specs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    output = ROOT / "data" / "v1-candidate"
    current_tools = [
        ToolRecord.model_validate(item)
        for item in json.loads((ROOT / "data" / "tools.json").read_text(encoding="utf-8"))
    ]
    tools = sorted([*current_tools, *NEW_TOOLS], key=lambda item: item.tool_id)
    if len(tools) != 40 or len({item.tool_id for item in tools}) != 40:
        raise RuntimeError("V1 candidate must contain 40 unique tools")
    corpus_hash = tool_corpus_sha256(tools)

    old_rows = [
        json.loads(line)
        for line in (ROOT / "data" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in old_rows:
        row["tool_corpus_hash"] = corpus_hash

    specs = make_specs()
    random.Random(SEED).shuffle(specs)
    split_slots = ["development"] * 60 + ["calibration"] * 70 + ["test"] * 70
    random.Random(SEED + 1).shuffle(split_slots)

    new_rows: list[dict[str, Any]] = []
    spec_by_query_id: dict[str, Spec] = {}
    for index, (spec, split) in enumerate(zip(specs, split_slots, strict=True), start=100):
        query_id = f"v1_q_{index:03d}"
        language = "en" if index % 2 == 0 else "zh"
        suffix = f" [V1 scenario {index}]" if language == "en" else f"（V1 场景 {index}）"
        should_abstain = spec.case_type in {"clarify", "no_match"}
        row = {
            "schema_version": "1.0",
            "query_id": query_id,
            "base_query_id": query_id,
            "text": (spec.en if language == "en" else spec.zh) + suffix,
            "language": language,
            "case_type": spec.case_type,
            "difficulty": (
                "easy"
                if spec.case_type == "clear"
                else "hard"
                if spec.case_type in {"no_match", "multi_valid"}
                else "medium"
            ),
            "relevant_tool_ids": list(spec.relevant),
            "should_abstain": should_abstain,
            "abstain_reason": spec.abstain_reason,
            "split": split,
            "label_rationale": spec.rationale,
            "candidate_conflict_note": spec.conflict,
            "tool_corpus_hash": corpus_hash,
            "perturbation_type": None,
        }
        QueryCase.model_validate(row)
        new_rows.append(row)
        spec_by_query_id[query_id] = spec

    base_rows = [*old_rows, *new_rows]
    if Counter(row["split"] for row in base_rows) != {
        "development": 100,
        "calibration": 100,
        "test": 100,
    }:
        raise RuntimeError("split allocation mismatch")
    if Counter(row["language"] for row in base_rows) != {"en": 150, "zh": 150}:
        raise RuntimeError("language allocation mismatch")

    test_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in new_rows:
        if row["split"] == "test":
            test_by_type[row["case_type"]].append(row)
    selected_bases: list[dict[str, Any]] = []
    for case_type in ("clear", "hard_negative", "multi_valid", "clarify", "no_match"):
        candidates = sorted(test_by_type[case_type], key=lambda row: row["query_id"])
        if len(candidates) < 6:
            raise RuntimeError(f"not enough new Test bases for {case_type}")
        selected_bases.extend(candidates[:6])

    variants: list[dict[str, Any]] = []
    for base in selected_bases:
        spec = spec_by_query_id[base["query_id"]]
        variant_text = {
            "paraphrase": (
                f"In other words, please: {spec.en}"
                if base["language"] == "en"
                else f"换一种说法：请{spec.zh}"
            ),
            "cross_language": spec.zh if base["language"] == "en" else spec.en,
            "noise": (
                f"umm... {spec.en} pls!!"
                if base["language"] == "en"
                else f"那个……{spec.zh}，谢啦！！"
            ),
        }
        for perturbation in ("paraphrase", "cross_language", "noise"):
            language = (
                ("zh" if base["language"] == "en" else "en")
                if perturbation == "cross_language"
                else base["language"]
            )
            row = {
                **base,
                "query_id": f"{base['query_id']}__{perturbation}",
                "base_query_id": base["query_id"],
                "text": f"{variant_text[perturbation]} [variant {base['query_id']}]",
                "language": language,
                "perturbation_type": perturbation,
            }
            QueryCase.model_validate(row)
            variants.append(row)

    query_rows = [*base_rows, *variants]
    annotation_rows = [
        {
            "schema_version": "1.0",
            "annotation_id": f"primary-{row['query_id']}",
            "annotator_id": "primary-v1-candidate",
            "query_id": row["query_id"],
            "relevant_tool_ids": row["relevant_tool_ids"],
            "should_abstain": row["should_abstain"],
            "abstain_reason": row["abstain_reason"],
            "label_rationale": row["label_rationale"],
            "candidate_conflict_note": row["candidate_conflict_note"],
            "tool_corpus_hash": corpus_hash,
        }
        for row in query_rows
    ]

    output.mkdir(parents=True, exist_ok=True)
    (output / "tools.json").write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in tools],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_jsonl(output / "queries.jsonl", query_rows)
    write_jsonl(output / "annotations" / "primary.jsonl", annotation_rows)

    source_rows = list(
        csv.DictReader((ROOT / "data" / "sources.csv").open(encoding="utf-8", newline=""))
    )
    for row in source_rows:
        if row["tool_id"].startswith("git.") and not row["original_identifier"].startswith("git_"):
            row["original_identifier"] = f"git_{row['original_identifier']}"
    for item in NEW_TOOLS:
        suffix = item.tool_id.split(".", 1)[1]
        original_identifier = (
            f"git_{suffix}" if item.domain == "git" else suffix.replace("get_sum", "get-sum")
        )
        source_rows.append(
            {
                "tool_id": item.tool_id,
                "source_url": item.source_url,
                "source_commit": COMMIT,
                "upstream_license": LICENSE,
                "access_date": ACCESS_DATE,
                "original_identifier": original_identifier,
                "transformation_notes": (
                    "Namespaced ID; concise description normalized; Zod/Pydantic input "
                    "converted to JSON Schema; no implementation copied."
                ),
                "redistribution_status": "permitted_with_license_and_attribution",
            }
        )
    with (output / "sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(source_rows, key=lambda row: row["tool_id"]))

    tools_by_id = {item.tool_id: item for item in tools}

    def stratum_domain(row: dict[str, Any]) -> str:
        relevant = row["relevant_tool_ids"]
        return tools_by_id[relevant[0]].domain if relevant else "no_match"

    def balanced_pick(cases: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        remaining = list(cases)
        random.Random(SEED + len(cases)).shuffle(remaining)
        selected: list[dict[str, Any]] = []
        domain_counts: Counter[str] = Counter()
        difficulty_counts: Counter[str] = Counter()
        while len(selected) < count:
            row = min(
                remaining,
                key=lambda item: (
                    domain_counts[stratum_domain(item)],
                    difficulty_counts[item["difficulty"]],
                    item["query_id"],
                ),
            )
            remaining.remove(row)
            selected.append(row)
            domain_counts[stratum_domain(row)] += 1
            difficulty_counts[row["difficulty"]] += 1
        return selected

    sample: list[dict[str, Any]] = []
    for case_type in ("clear", "hard_negative", "multi_valid", "clarify", "no_match"):
        cases = [row for row in base_rows if row["case_type"] == case_type]
        en = balanced_pick([row for row in cases if row["language"] == "en"], 10)
        zh = balanced_pick([row for row in cases if row["language"] == "zh"], 10)
        if len(en) != 10 or len(zh) != 10:
            raise RuntimeError(f"secondary sample is not bilingual for {case_type}")
        sample.extend([*en, *zh])
    sample.sort(key=lambda row: row["query_id"])
    assignment_fields = [
        "query_id",
        "text",
        "language",
        "relevant_tool_ids",
        "should_abstain",
        "abstain_reason",
        "label_rationale",
        "candidate_conflict_note",
        "annotator_id",
        "tool_corpus_hash",
    ]
    with (output / "annotations" / "secondary-assignment.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=assignment_fields)
        writer.writeheader()
        for row in sample:
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "text": row["text"],
                    "language": row["language"],
                    "relevant_tool_ids": "",
                    "should_abstain": "",
                    "abstain_reason": "",
                    "label_rationale": "",
                    "candidate_conflict_note": "",
                    "annotator_id": "",
                    "tool_corpus_hash": corpus_hash,
                }
            )

    manifest_fields = ["query_id", "language", "case_type", "difficulty", "stratum_domain"]
    with (output / "annotations" / "secondary-sample-manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for row in sample:
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "language": row["language"],
                    "case_type": row["case_type"],
                    "difficulty": row["difficulty"],
                    "stratum_domain": stratum_domain(row),
                }
            )

    recheck_rows = [
        row for row in base_rows if row["case_type"] in {"clarify", "hard_negative", "no_match"}
    ]
    with (output / "annotations" / "primary-recheck-assignment.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=assignment_fields)
        writer.writeheader()
        for row in sorted(recheck_rows, key=lambda item: item["query_id"]):
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "text": row["text"],
                    "language": row["language"],
                    "relevant_tool_ids": "",
                    "should_abstain": "",
                    "abstain_reason": "",
                    "label_rationale": "",
                    "candidate_conflict_note": "",
                    "annotator_id": "",
                    "tool_corpus_hash": corpus_hash,
                }
            )

    handoff = f"""# V1.0 annotation handoff

This directory is a **candidate**, not a released V1.0 dataset. Production-profile
structure validation passes, but independent human review is still missing.

## Secondary annotator package

Give the second human only:

- `tools.json`;
- `annotations/secondary-assignment.csv`; and
- `../../docs/annotation-guide.md` from the repository.

Do **not** give them `queries.jsonl`, `annotations/primary.jsonl`,
`secondary-sample-manifest.csv`, or prior benchmark outputs. The manifest contains
gold strata and is coordinator-only.

The annotator must fill every blank label field, use a stable non-primary
`annotator_id`, preserve `query_id` and this corpus hash, and separate multiple Tool
IDs with `|`:

`{corpus_hash}`

Allowed abstention reasons are blank for routeable cases, `ambiguous`,
`insufficient_information`, and `no_match`. A No-match record uses an empty relevant
set. Return the completed CSV without sorting or deleting rows.

Freeze the returned file before discussing disagreements:

```powershell
python scripts/process_v1_annotations.py prepare `
  --completed path/to/completed-secondary.csv
```

This writes immutable `secondary.jsonl`, `agreement.json`, and a disagreement-only
`adjudication-assignment.csv`. The command refuses to overwrite existing review
artifacts.

## Primary self-check

`annotations/primary-recheck-assignment.csv` contains all 180 Clarify, Hard Negative,
and No-match base cases without their existing labels. Complete it as a separate pass
without opening `primary.jsonl`, then compare and adjudicate changes explicitly.

```powershell
python scripts/process_v1_annotations.py recheck `
  --completed path/to/completed-primary-recheck.csv
```

After an adjudicator fills every blank adjudicated field in
`adjudication-assignment.csv`, freeze the final disagreement records with:

```powershell
python scripts/process_v1_annotations.py finalize `
  --completed path/to/completed-adjudication.csv
```

## Release gate

Do not rename this directory or promote its files into `data/` until:

1. all 100 secondary rows are complete and validated;
2. raw primary and secondary labels are frozen before adjudication;
3. route/abstain Cohen's Kappa and Tool-set Jaccard are calculated;
4. disagreement rows and adjudicator decisions are retained; and
5. any Test correction increments the dataset version and reruns every experiment.
"""
    (output / "ANNOTATION_HANDOFF.md").write_text(handoff, encoding="utf-8", newline="\n")

    dataset_card = f"""# V1.0 candidate Dataset Card

## Status

This directory is a **review candidate**, not the stable V1.0 benchmark. It passes
the production structural profile, but no independent secondary labels,
adjudications, agreement measurements, or frozen V1.0 benchmark results exist yet.
Do not cite candidate counts as a released human-reviewed dataset.

## Composition

| Item | Count |
|---|---:|
| Normalized MCP tools | 40 |
| Base queries | 300 |
| English / Chinese bases | 150 / 150 |
| Development / Calibration / Test bases | 100 / 100 / 100 |
| Clear / Hard Negative / Multi-valid / Clarify / No-match | 100 / 60 / 20 / 60 / 60 |
| Paraphrase / cross-language / noise variants | 30 / 30 / 30 |
| Total query records | 390 |

All variants share their base query's split. The immutable candidate Tool Corpus
hash is `{corpus_hash}`.

## Construction and labels

The first 100 bases come from V0.1. The additional 200 bases are synthetic candidate
records generated deterministically from project-authored intent templates and
provisional labels in `scripts/build_v1_candidate.py`. Scenario suffixes make record
identity explicit; they do not represent natural production traffic. The 90 paired
variants are also deterministic. These labels must be treated as provisional until
the documented human review is complete.

The secondary sample contains exactly 100 base queries: 10 English and 10 Chinese
examples from each of the five case types, selected while balancing Tool domain and
difficulty. Its gold strata are kept in a coordinator-only manifest and must not be
given to the blinded annotator. All 180 Clarify, Hard Negative, and No-match bases
also have a blank primary self-check assignment.

Follow `ANNOTATION_HANDOFF.md`. Raw primary, secondary, and recheck files are frozen
before discussion. Route/abstain agreement uses Cohen's Kappa; relevant Tool sets use
per-query and mean Jaccard. Constant-label Kappa is reported as unavailable. Label
disagreements retain both raw records and a separate adjudication record.

## Tool provenance and licensing

All normalized Tool records derive from the official
`modelcontextprotocol/servers` repository at commit `{COMMIT}`. `sources.csv`
records the exact source file, upstream identifier, access date, license transition,
and transformation for every Tool. No upstream implementation code is redistributed.

Project code is Apache-2.0. Original queries, provisional annotations,
documentation, and normalized descriptions are CC BY 4.0. Upstream schemas retain
their recorded Apache-2.0/MIT terms; consult `sources.csv` before redistribution.

## Intended use and limitations

This candidate evaluates single-step routing, clarification, and abstention over one
known Tool Corpus. It does not execute tools, generate arguments, measure downstream
task success, test unseen-Tool cold start, represent production traffic, or provide a
formal safety guarantee. The deliberately balanced stress-test priors differ from a
deployment; calibrated probabilities and thresholds can shift with class and query
distribution. Synthetic wording and repeated intent families can inflate apparent
regularity and must be disclosed in any derived report.

## Release gate

Stable V1.0 requires all of the following: complete 100-query independent review;
complete 180-query primary recheck; frozen raw labels; agreement report; explicit
adjudication of every change/disagreement; final corpus/query hashes; one frozen Test
evaluation per configuration; inspected manifests, metrics, error slices, policies,
figures, and reliability cards; and a clean CI/release rehearsal. Any correction to
frozen Test labels increments the dataset version and reruns every final experiment.
"""
    (output / "DATASET_CARD.md").write_text(
        dataset_card, encoding="utf-8", newline="\n"
    )

    summary = {
        "tool_count": len(tools),
        "base_query_count": len(base_rows),
        "variant_count": len(variants),
        "total_query_records": len(query_rows),
        "split_base_counts": dict(sorted(Counter(row["split"] for row in base_rows).items())),
        "language_base_counts": dict(sorted(Counter(row["language"] for row in base_rows).items())),
        "case_type_base_counts": dict(
            sorted(Counter(row["case_type"] for row in base_rows).items())
        ),
        "perturbation_counts": dict(
            sorted(Counter(row["perturbation_type"] for row in variants).items())
        ),
        "primary_annotation_count": len(annotation_rows),
        "secondary_assignment_count": len(sample),
        "secondary_sample_domain_counts": dict(
            sorted(Counter(stratum_domain(row) for row in sample).items())
        ),
        "primary_recheck_count": len(recheck_rows),
        "tool_corpus_hash": corpus_hash,
        "secondary_labels_present": False,
    }
    (output / "candidate-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()

"""
CLAUDE CODE - Agentic Coding Assistant

Claude Code is Anthropic's official CLI tool - an agentic coding assistant that runs
in your terminal and helps with the entire software development lifecycle.

KEY INSIGHT: Claude Code is an EFFORT MULTIPLIER
Not just for writing code - helps with every step of a project.

WHAT CLAUDE CODE IS:
- Terminal-based AI coding assistant
- Full file system access to your project
- Command execution capabilities
- Web search and documentation access
- Extensible via MCP servers
- Multi-instance support for parallel work

WHAT MAKES CLAUDE CODE DIFFERENT:
Unlike traditional coding assistants that just autocomplete or chat:
- AGENTIC: I can autonomously use tools to complete tasks
- PROACTIVE: I explore codebases, run tests, and verify my work
- ITERATIVE: I debug my own code and refine solutions
- CONTEXTUAL: I understand your full project, not just single files
- EXTENSIBLE: Connect MCP servers to add custom capabilities

THE FULL DEVELOPMENT LIFECYCLE:

1. DISCOVER
   - Explore codebase and history
   - Search documentation
   - Onboard to new projects

2. DESIGN
   - Plan projects
   - Develop technical specs
   - Define architecture

3. BUILD
   - Implement code
   - Write and execute tests
   - Create commits and PRs

4. DEPLOY
   - Automate CI/CD
   - Configure environments
   - Manage deployments

5. SUPPORT & SCALE
   - Debug errors
   - Large-scale refactors
   - Monitor usage and performance

COMMON WORKFLOW PATTERN:

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Feed Context    │ →  │ Plan Solution   │ →  │ Implement       │
│                 │    │                 │    │                 │
│ Find relevant   │    │ Tell Claude NOT │    │ Claude writes   │
│ files for a     │    │ to write code   │    │ code based on   │
│ feature         │    │ yet - just plan │    │ context & plan  │
└─────────────────┘    └─────────────────┘    └─────────────────┘

TIP: Consider asking Claude to plan WITHOUT writing code first.
This ensures alignment before implementation begins.
"""

# ============================================================================
# SETUP AND INSTALLATION
# ============================================================================
"""
PREREQUISITES:
- Node.js (check with: npm help)
- Anthropic API key

INSTALLATION:
npm install -g @anthropic-ai/claude-code

START CLAUDE CODE:
claude

(First run will prompt for login)

DOCUMENTATION:
Full setup guide: https://docs.anthropic.com

SYSTEM REQUIREMENTS:
- macOS, Linux, or Windows (WSL recommended)
- Terminal with color support
- Internet connection for API access
"""

# ============================================================================
# CORE CAPABILITIES
# ============================================================================
"""
WHAT CLAUDE CODE CAN DO:

1. FILE OPERATIONS
   - Search: Find files by name, pattern, or content
   - Read: Access any file in your project
   - Edit: Modify files with precise changes
   - Create: Write new files when needed
   - Delete: Remove files (with confirmation)

   I prefer editing existing files over creating new ones.
   I can work with any text-based file format.

2. TERMINAL ACCESS
   - Run commands: npm, git, python, docker, etc.
   - Execute tests: pytest, jest, cargo test
   - Build projects: make, npm build, cargo build
   - Install dependencies: npm install, pip install
   - Git operations: commit, push, branch, merge

   I verify command results and iterate if needed.

3. WEB ACCESS
   - Search documentation
   - Fetch code examples
   - Look up error messages
   - Check API references
   - Find best practices

   I cite sources and verify information.

4. MCP SERVER SUPPORT
   - Connect to external services
   - Custom tool integrations
   - Extend capabilities dynamically
   - Use community-built servers

   Built-in MCP client means instant extensibility.

WHAT I CANNOT DO:
✗ Access files outside your project directory
✗ Make network requests from your machine (but I can fetch web content)
✗ Interact with GUI applications (except via Computer Use)
✗ Access your personal data or credentials
✗ Execute privileged operations without confirmation
"""

# ============================================================================
# WORKING WITH CLAUDE CODE - BEST PRACTICES
# ============================================================================
"""
EFFECTIVE COLLABORATION PATTERNS:

1. PROVIDE CONTEXT UPFRONT
   Good: "Add authentication to this Express app. We use JWT tokens
          and store users in PostgreSQL."

   Better: "Add authentication to this Express app. We use JWT tokens
           and store users in PostgreSQL. Look at src/auth/login.js to see
           the existing pattern. Match that style."

2. USE THE THREE-PHASE WORKFLOW
   Phase 1: "Find all the files related to user authentication"
   Phase 2: "Plan how to add 2FA without breaking existing auth.
             Don't write code yet, just explain the approach."
   Phase 3: "Implement the plan you just described"

3. BE SPECIFIC ABOUT WHAT NOT TO DO
   - "Don't modify the database schema"
   - "Don't change the API contract"
   - "Keep the existing error handling pattern"
   - "Match the style of existing components"

4. LEVERAGE MY AUTONOMY
   Instead of: "Read file X, then read file Y, then check Z..."
   Just say: "Fix the authentication bug users are reporting"

   I'll explore, investigate, and solve it.

5. VERIFY MY WORK
   Ask me to:
   - Run tests after changes
   - Verify builds succeed
   - Check for regressions
   - Explain my reasoning

   I'm thorough, but confirmation is good practice.

6. ITERATE WITH FEEDBACK
   "That works, but can you make it more efficient?"
   "Good start, but use async/await instead of promises"
   "Almost there - handle the edge case where user is null"

7. USE CLEAR, DIRECT LANGUAGE
   Good: "Add error handling to the API endpoint"
   Avoid: "Maybe we should think about possibly adding some error stuff"

COMMUNICATION TIPS:

DO:
✓ Give me the full picture upfront
✓ Reference specific files, functions, or patterns
✓ Tell me your constraints and requirements
✓ Ask me to explain my approach before coding
✓ Request tests for new functionality
✓ Have me verify my changes work

DON'T:
✗ Micro-manage every step
✗ Be vague about requirements
✗ Assume I know undocumented conventions
✗ Skip the planning phase for complex tasks
✗ Forget to mention edge cases or constraints
"""

# ============================================================================
# MCP INTEGRATION - EXTENDING CLAUDE CODE
# ============================================================================
"""
CLAUDE CODE HAS A BUILT-IN MCP CLIENT

This means you can connect MCP servers to extend my capabilities.
No code changes needed - just configuration.

ABOUT MCP AND AAIF:

The Model Context Protocol (MCP) is now part of the Linux Foundation's
Agentic AI Foundation (AAIF) - a neutral, open governance structure for
agentic AI projects.

Major industry backing includes:
- AWS, Anthropic, Block, Bloomberg
- Cloudflare, Google, Microsoft, OpenAI

Over 10,000 MCP servers now exist in the rapidly growing ecosystem.

MCP provides a universal standard for connecting AI models to:
- External tools and services
- Data sources and databases
- Custom integrations

OFFICIAL MCP RESOURCES:

Main Website: https://modelcontextprotocol.io
GitHub Organization: https://github.com/modelcontextprotocol

Key Repositories:
- **Servers** (github.com/modelcontextprotocol/servers)
  Official collection of maintained server implementations
  Includes reference servers like: Everything, Fetch, Filesystem, Git, Memory

- **Registry** (github.com/modelcontextprotocol/registry)
  Community-driven registry for discovering MCP servers
  Explore with caution - quality varies

- **Inspector** (github.com/modelcontextprotocol/inspector)
  Visual testing tool for validating MCP server functionality
  Essential for development and debugging

- **Specification** (github.com/modelcontextprotocol/specification)
  Official protocol specification and documentation

CONFIGURING MCP SERVERS:

Claude Code looks for MCP server configuration in:
~/.claude/config.json

Example configuration:
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    },
    "custom-server": {
      "command": "python",
      "args": ["path/to/your/mcp_server.py"],
      "env": {
        "API_KEY": "your-api-key"
      }
    }
  }
}

ONCE CONFIGURED:
I automatically discover tools from all connected MCP servers.
You don't need to tell me which server to use - I figure it out.

FINDING MCP SERVERS:

1. **Official Servers Repository**
   Start here for vetted, maintained servers
   https://github.com/modelcontextprotocol/servers

2. **Community Registry**
   Explore community-built servers
   https://github.com/modelcontextprotocol/registry
   Note: Quality varies - review before using

3. **Build Your Own**
   See the MCP.py template in this repository for guidance
   Full documentation at modelcontextprotocol.io

THE POWER OF MCP:
- Instantly extend my capabilities
- No custom integration code needed
- Use community servers or build your own
- Industry-standard protocol backed by major players
"""

# ============================================================================
# ADVANCED PATTERN: PARALLELIZATION WITH GIT WORKTREES
# ============================================================================
"""
RUNNING MULTIPLE INSTANCES OF CLAUDE CODE

For large projects, you can run multiple Claude Code instances simultaneously
using Git worktrees. This allows parallel development on different features.

GIT WORKTREES EXPLAINED:
- Multiple working directories from one repository
- Each worktree can be on a different branch
- Share the same Git history
- Independent working state

SETUP WORKFLOW:

1. CREATE A WORKTREE
   Prompt: "Your task is to create a new worktree named 'feature_auth'
           in the .trees/feature_auth folder.

           Follow these steps:
           1. Check if .trees/feature_auth already exists. If so, stop
              and tell me it exists.
           2. Create a new git worktree in the .trees folder with the
              name feature_auth.
           3. Symlink the .venv folder into the worktree directory
           4. Launch a new VSCode editor instance in that directory
              by running the 'code' command"

2. OPEN SECOND CLAUDE CODE INSTANCE
   cd .trees/feature_auth
   claude

3. WORK IN PARALLEL
   - Main instance: Working on bug fixes
   - Second instance: Building new feature
   - Third instance: Refactoring old code

4. MERGE WHEN READY
   Each worktree can create PRs independently
   Merge through your normal Git workflow

BENEFITS:
✓ No branch switching disruption
✓ Different features progress simultaneously
✓ Each instance has full context of its branch
✓ Independent test runs
✓ Parallel CI/CD pipelines

TIPS:
- Symlink virtual environments (not tracked by Git)
- Use descriptive worktree names
- Clean up worktrees when features merge: git worktree remove feature_auth
- Consider custom slash commands for worktree management

CUSTOM SLASH COMMANDS:

You can create reusable commands for common workflows.
Store them in .claude/commands/ directory.

Example: /create-worktree command
Automates the worktree creation process above.

Example: /merge-feature command
Automates merging a feature branch with conflict resolution.
"""

# ============================================================================
# AUTOMATED DEBUGGING - PRODUCTION MONITORING
# ============================================================================
"""
PATTERN: AUTONOMOUS DEBUG & FIX WORKFLOW

Problem: Your app works perfectly in development but breaks in production.
Solution: Claude Code monitors production, identifies issues, and fixes them
         automatically while you sleep.

THE WORKFLOW:

1. GITHUB ACTION RUNS DAILY (typically early morning)
   - Scheduled via cron: "0 6 * * *"  # 6 AM daily
   - Triggered on error webhooks (optional)
   - Runs in isolated environment

2. CLAUDE QUERIES ERROR LOGS
   - CloudWatch, Sentry, Datadog, or custom logs
   - Last 24 hours of errors
   - Filtered by severity, service, or pattern

3. FILTER & DEDUPLICATE ERRORS
   - Group similar errors
   - Rank by frequency and impact
   - Fit within context window limits
   - Prioritize user-facing issues

4. CLAUDE ANALYZES AND FIXES
   - Reads error stack traces
   - Explores relevant code
   - Identifies root cause
   - Implements fix
   - Runs tests to verify

5. AUTO-COMMIT AND OPEN PR
   - Descriptive commit message with error details
   - PR includes error analysis and fix explanation
   - Links to error logs
   - Tags relevant team members

COMPONENTS NEEDED:

1. GitHub Actions Workflow
   - Runs Claude Code in CI environment
   - Accesses error monitoring service
   - Has write permissions to repository

2. Error Aggregation
   - API access to CloudWatch/Sentry/etc.
   - Filtering logic for relevance
   - Deduplication algorithm

3. Context Management
   - Keep errors within token limits
   - Prioritize by business impact
   - Include relevant code context

4. Git Automation
   - Branch creation
   - Commit with detailed message
   - PR creation with description
   - Notification to team

5. Review Process
   - Auto-merge for trivial fixes (typos, obvious bugs)
   - Require approval for risky changes
   - Run full test suite in PR
   - Security scanning

EXAMPLE GITHUB ACTION:

name: Automated Debug

on:
  schedule:
    - cron: '0 6 * * *'  # 6 AM daily
  workflow_dispatch:  # Manual trigger

jobs:
  debug-production-errors:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install Claude Code
        run: npm install -g @anthropic-ai/claude-code

      - name: Run Automated Debugging
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          CLOUDWATCH_ACCESS: ${{ secrets.CLOUDWATCH_ACCESS }}
        run: |
          claude --non-interactive "
          Your task is to fix production errors from the last 24 hours.

          1. Query CloudWatch for errors with severity HIGH or CRITICAL
          2. Filter and deduplicate to find the top 5 most impactful errors
          3. For each error:
             - Investigate the root cause
             - Implement a fix
             - Write/update tests
             - Verify the fix works
          4. Create commits with descriptive messages
          5. Open a PR with:
             - Summary of errors fixed
             - Links to CloudWatch logs
             - Test results
             - Risk assessment

          If you can't fix an error automatically, document why
          and include investigation notes in the PR.
          "

CUSTOMIZATION OPTIONS:

Error Sources:
- Sentry: Rich error context, user impact data
- Datadog: Distributed tracing, APM integration
- CloudWatch: AWS-native, log aggregation
- Rollbar: Detailed stack traces, occurrence tracking
- Custom logs: Elasticsearch, Splunk, etc.

Scheduling:
- Hourly: For high-traffic production apps
- Daily: Standard for most applications
- On-error webhooks: Real-time response
- Continuous: Monitor and fix in real-time

Scope:
- Specific services or microservices
- Error severity levels
- User-facing vs. internal errors
- Specific error patterns or types

Review Process:
- Auto-merge: Low-risk fixes (typos, null checks)
- Quick review: Medium-risk (logic fixes)
- Full review: High-risk (database, security)
- Staging deployment: Test before production

Notifications:
- Slack: Summary of fixes applied
- Email: Daily digest of PR activity
- PagerDuty: Critical errors needing human attention
- Dashboard: Metrics on fix success rate

THE POWER:
Wake up to:
- Production errors already analyzed
- Fixes implemented and tested
- PRs ready for review
- Detailed investigation notes

You review and merge. I do the grunt work.
"""

# ============================================================================
# COMPUTER USE - GUI INTERACTION CAPABILITIES
# ============================================================================
"""
WHAT IS COMPUTER USE?

Computer Use gives Claude Code the ability to interact with graphical
user interfaces - clicking, typing, scrolling, and observing like a human.
This is done safely in a sandbox.

CAPABILITIES:
- Open and navigate web browsers
- Click buttons, links, and UI elements
- Type into input fields
- Scroll and interact with pages
- Take screenshots
- Verify visual state
- Conduct QA testing
- Automate GUI workflows

HOW IT WORKS:

Behind the scenes, Computer Use is just another tool in my toolkit.
When you ask me to test a website or interact with a GUI:

1. I receive the request: "Test the login flow"
2. Tool use activates: computer_use tool
3. I break it down: Navigate → Click → Type → Verify
4. Actions execute: Mouse coordinates, keyboard input
5. I observe results: Screenshot analysis
6. I iterate: If something fails, I debug and retry

It's automatic - you don't need to think about tool schemas or
conversion logic. Just tell me what to do, and I figure out how.

EXAMPLE WORKFLOW: QA TESTING

Prompt:
"Your goal is to conduct QA testing on a React component
hosted at https://test-mentioner.vercel.app/

Testing process:
1. Open a new browser tab
2. Navigate to https://test-mentioner.vercel.app/
3. Execute the test cases below one by one
4. After completing all tests, write a concise report

Test cases:
1. Typing 'Did you read @' should display autocomplete options
2. Typing 'Did you read @' then pressing enter should add '@document.pdf'
3. After adding '@document.pdf', pressing backspace should show
   autocomplete options directly below the text, not elsewhere on the page"

I will:
- Open the browser
- Navigate to the URL
- Execute each test case
- Take screenshots showing results
- Document any bugs or issues
- Provide a detailed QA report

DOCKER CONTAINER SETUP:

Computer Use runs in an isolated Docker container for safety.
Anthropic provides a preconfigured container with:
- Desktop environment (Xfce)
- Web browser (Firefox)
- VNC server for remote viewing
- All necessary tools

SETUP COMMAND:

export ANTHROPIC_API_KEY="your_api_key"

docker run \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $HOME/.anthropic:/home/computeruse/.anthropic \
  -p 5900:5900 \
  -p 8501:8501 \
  -p 6080:6080 \
  -p 8080:8080 \
  -it ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest

PORTS EXPLAINED:
- 5900: VNC server (remote desktop viewing)
- 8501: Streamlit web interface
- 6080: noVNC web-based VNC client
- 8080: HTTP server for web apps

ACCESS METHODS:
- Web UI: Open http://localhost:8501
- VNC viewer: Connect to localhost:5900
- Browser VNC: Open http://localhost:6080

SAFETY & ISOLATION:

The Docker container provides isolation:
✓ Separate from your main system
✓ No access to personal files
✓ Can't affect your actual desktop
✓ Safe for testing potentially buggy apps
✓ Easy to reset/restart

IMPORTANT:
Computer Use is still in beta. It works well for:
✓ QA testing web applications
✓ Visual regression testing
✓ Troubleshooting UI bugs
✓ Verifying responsive design
✓ Accessibility testing

But may struggle with:
✗ Complex multi-step workflows
✗ Precise pixel-perfect interactions
✗ Very fast-moving animations
✗ Canvas or WebGL applications
✗ Non-standard UI frameworks

USE CASES:

1. QA Testing
   "Test the checkout flow on staging and report any issues"
   I'll click through the entire flow, fill forms, and verify results.

2. Visual Regression Testing
   "Compare the homepage design to the Figma mockup"
   I'll take screenshots and identify visual differences.

3. Cross-Browser Testing
   "Test this feature in Firefox and report compatibility issues"
   I'll test and document browser-specific problems.

4. Accessibility Testing
   "Verify the app is navigable with keyboard only"
   I'll test without using the mouse and report issues.

5. Bug Reproduction
   "Try to reproduce the bug users reported about the dropdown"
   I'll follow repro steps and document what happens.

6. User Flow Validation
   "Walk through the user onboarding flow and suggest improvements"
   I'll experience it like a user and provide UX feedback.

TIPS FOR COMPUTER USE:

1. Be Specific
   Good: "Click the blue 'Sign In' button in the top right"
   Vague: "Sign in somehow"

2. Break Down Complex Flows
   Instead of: "Test everything"
   Better: "Test the login flow, then test the dashboard"

3. Expect Iteration
   I might need to retry actions or adjust approach.
   Computer Use is autonomous but not perfect.

4. Provide Expected Results
   "After clicking, you should see a success message"
   This helps me verify correct behavior.

5. Use for What Humans Do
   If a human tester would do it, I can probably do it.
   If it needs specialized tools, use other approaches.
"""

# ============================================================================
# COMMON USE CASES & EXAMPLES
# ============================================================================
"""
DISCOVER PHASE:

"Explore this codebase and explain how authentication works"
→ I'll read relevant files, trace the flow, and explain

"Find all places where we make database queries"
→ I'll search for DB calls and list them with context

"What external APIs does this project depend on?"
→ I'll check imports, configs, and code for API usage

DESIGN PHASE:

"Plan how to add real-time notifications without breaking existing code"
→ I'll analyze the codebase and propose an approach (no code yet)

"Design a database schema for a multi-tenant SaaS application"
→ I'll create a schema design with explanation

"What's the best way to add caching to this API?"
→ I'll evaluate options and recommend an approach

BUILD PHASE:

"Implement the authentication plan we discussed"
→ I'll write the code based on our agreed approach

"Add unit tests for the payment processing module"
→ I'll write comprehensive tests with good coverage

"Refactor the UserService class to use dependency injection"
→ I'll refactor while maintaining existing behavior

DEPLOY PHASE:

"Set up a GitHub Actions workflow for CI/CD"
→ I'll create a workflow file with build, test, deploy

"Write a Dockerfile for this Express app"
→ I'll create an optimized Docker setup

"Add environment variable configuration for staging and prod"
→ I'll implement proper config management

SUPPORT & SCALE PHASE:

"Debug why the API is slow for large result sets"
→ I'll investigate, find bottlenecks, and optimize

"Refactor the monolith to extract the auth service"
→ I'll carefully separate concerns and maintain compatibility

"Find and fix all TODO comments in the codebase"
→ I'll locate TODOs, assess them, and resolve where appropriate
"""

# ============================================================================
# BEST PRACTICES SUMMARY
# ============================================================================
"""
WORKING EFFECTIVELY WITH CLAUDE CODE:

CONTEXT IS KING:
✓ Tell me about your project structure upfront
✓ Explain conventions and patterns you follow
✓ Reference specific files, functions, or modules
✓ Share constraints and requirements early
✓ Mention edge cases and error scenarios

THE THREE-PHASE WORKFLOW:
1. Feed Context: Show me relevant files and patterns
2. Plan Solution: Ask me to design (not implement yet)
3. Implement: Have me write code based on the plan

COMMUNICATION:
✓ Be specific and direct
✓ One task at a time (but I can handle complex tasks)
✓ Tell me what NOT to do (constraints matter)
✓ Ask me to explain my reasoning
✓ Request verification (run tests, check builds)

LEVERAGE MY AUTONOMY:
✓ Let me explore and investigate
✓ Trust me to figure out the details
✓ Don't micro-manage every step
✓ I'll ask if I need clarification

ITERATE AND REFINE:
✓ Give feedback on my work
✓ Ask for improvements or alternatives
✓ Have me optimize or refactor
✓ Build incrementally with verification

EXTEND MY CAPABILITIES:
✓ Connect MCP servers for your tools
✓ Use Computer Use for GUI testing
✓ Run multiple instances for parallel work
✓ Automate repetitive workflows

VERIFY MY WORK:
✓ Ask me to run tests after changes
✓ Have me check builds succeed
✓ Request explanation of changes
✓ Review before merging (I'm good, not perfect)
"""

# ============================================================================
# PATTERN SUMMARY - QUICK REFERENCE
# ============================================================================
"""
DEVELOPMENT LIFECYCLE PATTERNS:

DISCOVER:
- "Explore the codebase and explain [feature]"
- "Find all [pattern] in the project"
- "What does [module] do and how does it work?"

DESIGN:
- "Plan how to add [feature] - don't code yet"
- "Design [architecture component] and explain tradeoffs"
- "What's the best approach for [problem]?"

BUILD:
- "Implement [feature] following [pattern]"
- "Add [tests] for [module]"
- "Refactor [code] to [improvement]"

DEPLOY:
- "Set up [CI/CD platform] workflow"
- "Create [Docker/Kubernetes] configuration"
- "Add [environment] configuration"

SUPPORT:
- "Debug why [issue] is happening"
- "Fix [bug] reported by users"
- "Optimize [slow operation]"

ADVANCED PATTERNS:

PARALLELIZATION:
- Create worktrees for parallel development
- Run multiple Claude Code instances
- Independent feature branches

AUTOMATION:
- GitHub Actions for auto-debugging
- Scheduled error monitoring and fixing
- Auto-PR creation with fixes

COMPUTER USE:
- QA testing web applications
- Visual regression testing
- User flow validation

MCP INTEGRATION:
- Connect to external services
- Extend capabilities dynamically
- Use community servers

TROUBLESHOOTING:

CLAUDE NOT FINDING FILES:
→ "Search for files matching [pattern]"
→ Check working directory: pwd

CLAUDE MAKING WRONG ASSUMPTIONS:
→ Provide more context upfront
→ Reference specific examples: "Like in [file]"

CHANGES NOT WORKING:
→ "Run the tests and verify"
→ "Check if the build succeeds"

NEED TO UNDO CHANGES:
→ "Revert the changes to [file]"
→ Git is your friend: git restore [file]

TOO MANY CHANGES AT ONCE:
→ "Let's do this incrementally"
→ "First just [step 1], then we'll do [step 2]"

UNCLEAR REQUIREMENTS:
→ I'll ask clarifying questions
→ Consider the plan phase: "Design first, code later"
"""

# ============================================================================
# EXAMPLE USAGE - REAL WORKFLOWS
# ============================================================================

if __name__ == "__main__":
    """
    This isn't executable code - it's documentation of how to use Claude Code.

    GETTING STARTED:

    1. Install: npm install -g @anthropic-ai/claude-code
    2. Run: claude
    3. Login when prompted
    4. Start collaborating!

    EXAMPLE SESSION:

    You: "I want to add user authentication to this Express app"

    Claude: "I'll help you add authentication. First, let me explore
             the codebase to understand the current structure."
             [Reads relevant files]
             "I can see you're using Express with PostgreSQL. Before I
             start coding, let me plan the approach:

             1. Add bcrypt for password hashing
             2. Create auth middleware for protected routes
             3. Add login/register endpoints
             4. Use JWT tokens for sessions
             5. Store tokens in HTTP-only cookies

             Does this approach work for you, or would you prefer
             something different?"

    You: "That sounds good, but use sessions instead of JWT"

    Claude: "Got it, I'll use express-session instead. Let me implement this."
             [Implements authentication]
             [Runs tests]
             "I've added authentication with sessions. All tests pass.
             Here's what I did:
             - Added session middleware in app.js
             - Created auth routes in routes/auth.js
             - Added password hashing with bcrypt
             - Created auth middleware for protected routes
             - Updated user model with password field
             - Added tests for all auth endpoints

             You can test it by running: npm test"

    NEXT STEPS:
    - Read the full docs: https://docs.anthropic.com
    - Join the community: Discord, GitHub Discussions
    - Try the examples above
    - Connect MCP servers for your workflow
    - Explore Computer Use for testing

    Happy coding! 🚀
    """
    print(__doc__)

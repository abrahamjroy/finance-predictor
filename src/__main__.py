"""
Expose MCP server as a runnable module.

Usage:
    python -m src.mcp_server
"""

from .mcp_server import main

if __name__ == "__main__":
    main()

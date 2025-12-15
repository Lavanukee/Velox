"""
Velox Chat Tools - Web Search and Code Execution

These tools can be called by the LLM during chat to perform actions.
"""

import json
import sys
import urllib.request
import urllib.parse
import subprocess
import tempfile
import os
import importlib.util
from typing import Any, Dict


def web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo Instant Answers API.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        Dict with search results
    """
    try:
        # DuckDuckGo Instant Answers API (no API key needed)
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Velox/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        results = []
        
        # Abstract (main answer)
        if data.get('Abstract'):
            results.append({
                'title': data.get('Heading', 'Answer'),
                'snippet': data['Abstract'],
                'url': data.get('AbstractURL', '')
            })
        
        # Related topics
        for topic in data.get('RelatedTopics', [])[:max_results]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append({
                    'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                    'snippet': topic.get('Text', ''),
                    'url': topic.get('FirstURL', '')
                })
        
        if not results:
            # If no instant answer, return a generic message
            return {
                'success': True,
                'query': query,
                'results': [],
                'message': f"No instant answers found for '{query}'. The model should use its knowledge or ask the user to try a different query."
            }
        
        return {
            'success': True,
            'query': query,
            'results': results[:max_results]
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'query': query
        }


def execute_python_code(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute Python code in a sandboxed environment.
    
    WARNING: This is a basic sandbox. For production, use proper isolation.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Dict with execution results
    """
    # List of forbidden imports/operations for basic safety
    FORBIDDEN = [
        'import os', 'from os', 'os.system', 'os.popen',
        'import subprocess', 'from subprocess',
        'import shutil', 'from shutil',
        '__import__', 'eval(', 'exec(',
        'open(', 'file(',
        'import socket', 'from socket',
        'import http', 'from http',
        'import urllib', 'from urllib',
        'import requests', 'from requests',
        'import sys', 'from sys',
        'sys.exit', 'exit(', 'quit(',
        'import ctypes', 'from ctypes',
        'import win32', 'from win32',
        'import _', 'from _',
    ]
    
    # Check for forbidden patterns
    code_lower = code.lower()
    for forbidden in FORBIDDEN:
        if forbidden.lower() in code_lower:
            return {
                'success': False,
                'error': f"Forbidden operation detected: {forbidden}",
                'stdout': '',
                'stderr': ''
            }
    
    # Create a temporary file for the code
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Wrap code to capture output
            wrapped_code = f'''
import json
import traceback

try:
    # User code
{chr(10).join("    " + line for line in code.split(chr(10)))}
except Exception as e:
    print(f"Error: {{e}}")
'''
            f.write(wrapped_code)
            temp_file = f.name
        
        # Execute with timeout
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir()
        )
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip(),
            'return_code': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': f'Execution timed out after {timeout} seconds',
            'stdout': '',
            'stderr': ''
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'stdout': '',
            'stderr': ''
        }
    finally:
        # Clean up temp file
        try:
            if 'temp_file' in locals():
                os.unlink(temp_file)
        except:
            pass


# Tool definitions for LLM
TOOL_DEFINITIONS = {
    'web_search': {
        'name': 'web_search',
        'description': 'Search the web for information using DuckDuckGo',
        'parameters': {
            'query': {
                'type': 'string',
                'description': 'The search query',
                'required': True
            },
            'max_results': {
                'type': 'integer',
                'description': 'Maximum number of results (default: 3)',
                'required': False
            }
        }
    },
    'execute_python_code': {
        'name': 'execute_python_code',
        'description': 'Execute Python code to perform calculations or data processing',
        'parameters': {
            'code': {
                'type': 'string',
                'description': 'Python code to execute',
                'required': True
            }
        }
    }
}


def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Execute a tool by name with given arguments."""
    tools = {
        'web_search': web_search,
        'execute_python_code': execute_python_code
    }

    # Load custom tools
    custom_dir = os.path.join(os.path.dirname(__file__), 'custom_tools')
    if os.path.exists(custom_dir):
        sys.path.append(custom_dir)
        for filename in os.listdir(custom_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                try:
                    module_name = filename[:-3]
                    spec = importlib.util.spec_from_file_location(module_name, os.path.join(custom_dir, filename))
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Register tool if it has a run function
                        if hasattr(module, 'run'):
                            tools[module_name] = module.run
                        # Register definition if present (for future UI usage, though purely backend for now)
                        if hasattr(module, 'TOOL_DEF'):
                             TOOL_DEFINITIONS[module_name] = module.TOOL_DEF
                except Exception as e:
                    print(f"Error loading custom tool {filename}: {e}", file=sys.stderr)

    if tool_name not in tools:
        return {'success': False, 'error': f'Unknown tool: {tool_name}'}
    
    return tools[tool_name](**kwargs)


if __name__ == '__main__':
    # CLI interface for testing
    if len(sys.argv) < 2:
        print("Usage: python tools.py <tool_name> <json_args>")
        print("Tools: web_search, execute_python_code")
        sys.exit(1)
    
    tool_name = sys.argv[1]
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    
    result = execute_tool(tool_name, **args)
    print(json.dumps(result, indent=2))

"""
Converts REPORT.md to a beautifully styled, standalone REPORT.html
for instant viewing in any browser directly from the Windows file manager.
"""
import markdown
import os

def generate_html_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(base_dir, "REPORT.md")
    html_path = os.path.join(base_dir, "REPORT.html")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert markdown to HTML with tables and fenced code blocks
    html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'toc'])

    styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Amazon Customer Support AI Agent - Engineering & Evaluation Report</title>
    <style>
        :root {{
            --primary: #232f3e;
            --accent: #ff9900;
            --text: #1a1a1a;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --success: #16a34a;
            --danger: #dc2626;
            --warning: #d97706;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text);
            background-color: var(--bg);
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 50px 60px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border: 1px solid var(--border);
        }}
        h1 {{
            color: var(--primary);
            border-bottom: 3px solid var(--accent);
            padding-bottom: 12px;
            margin-top: 0;
            font-size: 2.2rem;
        }}
        h2 {{
            color: var(--primary);
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
            margin-top: 36px;
            font-size: 1.5rem;
        }}
        h3 {{
            color: #334155;
            margin-top: 24px;
            font-size: 1.2rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 24px 0;
            font-size: 0.95rem;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border: 1px solid var(--border);
        }}
        th {{
            background-color: #f1f5f9;
            color: var(--primary);
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        code {{
            background: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: Consolas, Monaco, "Courier New", monospace;
            font-size: 0.9em;
            color: #0f172a;
        }}
        pre {{
            background: #0f172a;
            color: #f8fafc;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: Consolas, Monaco, "Courier New", monospace;
            font-size: 0.9em;
        }}
        pre code {{
            background: transparent;
            color: inherit;
            padding: 0;
        }}
        blockquote {{
            border-left: 4px solid var(--accent);
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #fffbeb;
            color: #92400e;
            border-radius: 0 8px 8px 0;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: var(--border);
            margin: 40px 0;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-warning {{ background: #fef3c7; color: #92400e; }}
        .badge-danger {{ background: #fee2e2; color: #991b1b; }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)
    print(f"Standalone HTML report generated at {html_path}")

if __name__ == "__main__":
    generate_html_report()

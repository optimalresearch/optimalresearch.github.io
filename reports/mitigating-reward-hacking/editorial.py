"""Apply reviewed trace introductions without changing recorded agent messages.

The same transformation handles preserved HTML and newly rendered traces.
Only the summary section changes; source links point into the existing trace.
"""
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def revise(page: str, slug: str) -> str:
    entries = json.loads((HERE / 'data/trace_editorial.json').read_text())
    entry = entries.get(slug)
    if entry is None:
        return page
    match = re.search(r'<section class="ch" id="final">[\s\S]*?</section>', page)
    if match is None:
        raise ValueError(f'{slug}: missing trace summary')
    original = match.group()
    # Keep links discovered from the original summary, or from an earlier edit.
    links = {}
    for block in re.findall(r'<(?:li|tr)\b[^>]*>[\s\S]*?</(?:li|tr)>', original):
        for item in entry['process']:
            if item['iteration'] in html.unescape(block):
                a = re.search(r'<a href="(#[^"]+)">([^<]+)</a>', block)
                if a:
                    links[item['iteration']] = (a[1], a[2])
    intro = re.search(r'<div class="submission-summary">\s*<p>[\s\S]*?</p>', original)
    if intro is None:
        raise ValueError(f'{slug}: missing submission paragraph')
    version = re.search(r'<code>(.*?)</code>', intro.group())
    if version is None:
        raise ValueError(f'{slug}: missing submitted version')
    summary = original[:intro.start()] + (
        '<div class="submission-summary"><p>' + html.escape(entry['method']) +
        f' Submitted version: <code>{version[1]}</code>.</p>'
    ) + original[intro.end():]
    items = []
    for item in entry['process']:
        iteration = item['iteration']
        target = links.get(iteration)
        if target is None:
            # Some rejected candidates have no dev row. Link their recorded command.
            trace = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', page[match.end():])
            position = trace.find(html.escape(iteration))
            anchors = re.findall(r'id="(s[0-9]+)"', trace[:position]) if position >= 0 else []
            if anchors:
                target = ('#' + anchors[-1], iteration)
            else:
                raise ValueError(f'{slug}: no evidence link for {iteration}')
        if f'id="{target[0][1:]}"' not in page:
            raise ValueError(f'{slug}: broken evidence link {target[0]}')
        items.append('<li><strong>' + html.escape(item['lead']) + '</strong> ' +
                     html.escape(item['text']) + ' <a href="' + target[0] + '">' +
                     html.escape(iteration) + '</a></li>')
    summary = re.sub(r'<h3>Scientific process</h3>[\s\S]*?(?=<h3>)',
                     '<h3>Scientific process</h3><ul>' + ''.join(items) + '</ul>', summary, count=1)
    summary = summary.replace('How the dev scores improved', 'Dev evaluation results')
    summary = summary.replace("The checks are numbered in the order they finished.", "We number checks in completion order.")
    # Explain candidate changes in plain language; keep the experiment identifiers.
    for iteration, description in entry.get('checks', {}).items():
        pattern = r'<td>' + re.escape(html.escape(iteration)) + r', [^<]*</td>'
        def describe(check):
            submitted = ' (submitted)' if '(submitted)' in check.group() else ''
            return '<td>' + html.escape(iteration + ', ' + description + submitted) + '</td>'
        summary = re.sub(pattern, describe, summary)
    return page[:match.start()] + summary + page[match.end():]


def refresh(folder: Path) -> None:
    for path in folder.glob('*.html'):
        before = path.read_text()
        after = revise(before, path.stem)
        if after != before:
            path.write_text(after)

"""Local chat page for the deep research agent.

Open http://127.0.0.1:7860 after starting this file. Type a ticker or a
research question; the reply is the markdown report from research_agent.
"""

from __future__ import annotations

try:
    import gradio as gr
except ImportError as e:
    raise ImportError(
        'Please install gradio with `py -3 -m pip install "gradio>=6.7.0"`.'
    ) from e

from research_agent import render_report, research


async def answer(message: str, history: list[dict]):
    """Run one research report and stream a status line, then the report."""
    del history
    query = (message or '').strip()
    if not query:
        yield 'Enter a ticker such as NVDA, or a research topic.'
        return
    yield 'Researching. This usually takes a few minutes while sources are searched and read.'
    try:
        report = await research(query)
    except Exception as exc:
        yield f'Research failed: {exc}'
        return
    yield render_report(report)


demo = gr.ChatInterface(
    fn=answer,
    title='Pydantic AI Deep Research Agent',
    description='Enter a stock ticker (e.g., NVDA) or a research topic to generate a detailed report.',
    examples=['NVDA', 'TSLA', 'solid-state battery manufacturing'],
    run_examples_on_click=False,
    textbox=gr.Textbox(
        placeholder='Enter a stock ticker (e.g., NVDA) or a research topic',
        lines=1,
        scale=7,
    ),
)


def launch() -> None:
    """Bind 7860, or the next free port if a previous chat page is still running."""
    for port in range(7860, 7870):
        try:
            demo.launch(server_name='127.0.0.1', server_port=port)
            return
        except OSError:
            continue
    raise SystemExit('No free port between 7860 and 7869.')


if __name__ == '__main__':
    launch()

"""4. Dashboard: every page renders, with and without sidebar filters (Streamlit's AppTest, offline chat)."""
import pytest
from streamlit.testing.v1 import AppTest

from config import ROOT

APP = ROOT / "app" / "app.py"
PAGES = ["page_portfolio", "page_segments", "page_account", "page_chat", "page_model", "page_anomalies"]

# AppTest can't follow st.navigation to another page, so this runs the real app with navigation pointed at
# the page under test. The chat is forced offline so the tests never call the LLM.
WRAPPER = """
import runpy, sys
import streamlit as st
from types import SimpleNamespace
sys.path.insert(0, {src!r})
import chat
chat.api_key = lambda: None
st.navigation = lambda groups, **kw: SimpleNamespace(run=next(
    p._page for pages in groups.values() for p in pages if p._page.__name__ == {page!r}))
runpy.run_path({app!r}, run_name="__main__")
"""


def run_page(page, **filters):
    at = AppTest.from_string(WRAPPER.format(src=str(ROOT / "src"), page=page, app=str(APP)), default_timeout=120)
    for column, values in filters.items():
        at.session_state[f"filter_{column}"] = values
    return at.run()


def test_app_starts_on_the_portfolio_view():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "Portfolio view"
    assert {m.label: m.value for m in at.metric}["Accounts"] == "14,685"


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("filters", [{}, {"region": ["South"]}, {"loan_type": ["Mortgage"], "age_band": ["21-25"]}])
def test_every_page_renders(page, filters):
    at = run_page(page, **filters)
    assert not at.exception, [e.message for e in at.exception]
    assert at.title


def test_filters_that_match_nothing_stop_with_a_message():
    at = run_page("page_portfolio", region=["South"], age_band=["56+"], income_band=["<30k"], loan_type=["Mortgage"])
    assert not at.exception
    assert any("No accounts match" in w.value for w in at.warning)


def test_the_briefs_example_question_is_answered_on_the_page():
    at = run_page("page_chat")
    at = at.button(key="suggest_0").click().run()
    assert not at.exception
    answer = at.chat_message[1].markdown[0].value
    assert "2023Q4" in answer and "%" in answer
    assert any("numbers were found in the data" in c.value for c in at.caption)

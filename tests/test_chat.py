"""5. AI feature: the chat's tools, its offline answers and the number check. No network calls."""
import pytest

import chat

TOOL_ARGUMENTS = {
    "get_overview": {}, "segment_breakdown": {"dimension": "loan_type"}, "highest_default_segments": {},
    "cross_tab": {"row_dimension": "region", "column_dimension": "loan_type"}, "vintage_trend": {},
    "account_lookup": {"account_id": "ACC203222"}, "list_accounts": {}, "model_info": {},
}


@pytest.mark.parametrize("name", list(chat.TOOLS))
def test_every_tool_runs_on_the_book_and_on_a_filtered_view(book, name):
    assert "error" not in chat.run_tool(book, name, TOOL_ARGUMENTS[name])
    view = book[book.region.eq("South")]
    arguments = {} if name == "account_lookup" else TOOL_ARGUMENTS[name]
    if name != "account_lookup":
        assert "error" not in chat.run_tool(view, name, arguments)


def test_years_are_labelled_as_years(book):
    periods = chat.run_tool(book, "vintage_trend", {})["periods"]
    assert [p["period"] for p in periods] == ["2018", "2019", "2020", "2021", "2022", "2023"]


@pytest.mark.parametrize("question", chat.SUGGESTED_QUESTIONS)
def test_every_suggested_question_is_answered_offline_and_checks_out(book, question):
    result = chat.offline_answer(book, question, reason="test")
    assert result["mode"] == "offline" and result["tool_calls"]
    assert result["verification"]["status"] == "verified", result


def test_offline_answers_respect_every_part_of_the_question(book):
    unemployed = chat.offline_answer(book, "Compare default rates across income bands for unemployed customers.", "t")
    assert unemployed["tool_calls"][0]["arguments"]["filters"] == {"employment_type": ["Unemployed"]}
    combination = chat.offline_answer(book, "Which region and loan type combination is riskiest?", "t")
    assert combination["tool_calls"][0]["tool"] == "cross_tab"
    assert "South Mortgage" in combination["answer"]
    exposure = chat.offline_answer(book, chat.SUGGESTED_QUESTIONS[1], "t")["answer"]
    assert "Mortgage has the highest default rate" in exposure and "of exposure" in exposure
    quarter = chat.offline_answer(book, chat.SUGGESTED_QUESTIONS[0], "t")
    assert quarter["tool_calls"][0]["arguments"]["filters"] == {"origination_quarter": ["2023Q4"]}


@pytest.mark.parametrize("rows", [0, 1, 40])
def test_offline_answers_survive_tiny_filtered_views(book, rows):
    for question in chat.SUGGESTED_QUESTIONS:
        answer = chat.offline_answer(book.iloc[:rows], question, "test")["answer"]
        assert answer and "None" not in answer and "nan" not in answer.lower()


@pytest.fixture(scope="module")
def loan_types(book):
    return [{"tool": "segment_breakdown", "arguments": {"dimension": "loan_type"},
             "result": chat.run_tool(book, "segment_breakdown", {"dimension": "loan_type"})},
            {"tool": "segment_breakdown", "arguments": {"dimension": "region"},
             "result": chat.run_tool(book, "segment_breakdown", {"dimension": "region"})}]


@pytest.mark.parametrize("answer, question, status", [
    ("Mortgages default most, at 24.7% across 1,432 accounts, holding 52.3% of exposure.", "", "verified"),
    ("Mortgages default most, at 31.2% across 1,432 accounts.", "", "check"),          # invented figure
    ("South has a default rate of 24.7%.", "", "check"),                                 # real figure, wrong segment
    ("South has a default rate of 15.0%, Central 16.9%.", "", "check"),                  # swapped within a sentence
    ("Central and South default at 15.0% and 16.9%, respectively.", "", "verified"),
    ("The default rate is 99%.", "Is the default rate 99%?", "check"),                    # echoed from the question
    ("There are 9 defaults in South.", "", "check"),                                      # small counts are checked too
    ("Across the top 5 products, Mortgage leads at 24.7%.", "", "verified"),             # "top 5" is not a data claim
])
def test_number_check(loan_types, answer, question, status):
    assert chat.verify(answer, loan_types, question)["status"] == status


def test_number_check_reports_where_each_number_came_from(loan_types):
    v = chat.verify("Mortgages default at 24.7%.", loan_types)
    assert v["sources"] == [{"number": "24.7%", "source": "segment_breakdown · mortgage · default_rate_pct"}]
    echoed = chat.verify("The default rate is 99%.", [], "Is the default rate 99%?")
    assert echoed["from_question"] == ["99%"]

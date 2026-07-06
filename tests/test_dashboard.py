from tests.test_server import make_client, tool_use_response


def test_dashboard_serves_html_with_panels() -> None:
    client = make_client([tool_use_response("q")])
    html = client.get("/").text
    for anchor in ("id=\"spend\"", "id=\"runs\"", "id=\"incidents\"", "EventSource",
                   "calls_per_minute"):  # rate chart backfills from status history
        assert anchor in html

from pathlib import Path

from seats import parse_seat_matrix_html


def test_parse_seat_matrix_html_normalizes_category_columns(tmp_path: Path):
    html = """
    <table>
      <tr>
        <th>Institute Name</th><th>Program Name</th><th>Quota</th><th>Seat Pool</th>
        <th>OPEN</th><th>OPEN-PwD</th><th>GEN-EWS</th><th>OBC-NCL-PwD</th>
      </tr>
      <tr>
        <td>Test Institute</td><td>Computer Science</td><td>AI</td><td>Gender-Neutral</td>
        <td>10</td><td>1</td><td>2</td><td>3</td>
      </tr>
    </table>
    """
    path = tmp_path / "2026_seat_matrix.html"
    path.write_text(html)

    seats = parse_seat_matrix_html(str(path))

    categories = set(seats["category"].tolist())
    assert {"OPEN", "OPEN (PwD)", "EWS", "OBC-NCL (PwD)"} == categories
    assert int(seats.loc[seats["category"] == "EWS", "seats"].iloc[0]) == 2

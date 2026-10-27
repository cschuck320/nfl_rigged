import gathering as gat
import summaries as s
import vegas_impact as vi


def main():
    start_year = 2016
    end_year = 2026
    df = gat.get_pbp(start_year, end_year)
    fig_facet, fig_all, fig_key = s.penalties_per_quarter_yearly(df, start_year, end_year)
    fig_vegas = vi.key_flags_benefiting_trailing_teams(df, start_year, end_year)

    # Combine all figures into a single HTML file
    file_path_name = "data/all_data.html"
    with open(file_path_name, "w", encoding="utf-8") as f:
        f.write("<html><head><title>NFL Penalty Analysis</title></head><body>\n")
        f.write(fig_facet.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("<hr>\n")
        f.write(fig_all.to_html(full_html=False, include_plotlyjs=False))
        f.write("<hr>\n")
        f.write(fig_key.to_html(full_html=False, include_plotlyjs=False))
        f.write("<hr>\n")
        f.write(fig_vegas.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("</body></html>\n")

    print(f"\nSaved combined charts to '{file_path_name}'")


if __name__ == "__main__":
    main()

import { fireEvent, render, screen } from "@testing-library/react";
import { AnswerPanel, parseAnswer } from "./AnswerPanel";
import type { AskResponse } from "./api";

const result: AskResponse = {
  question: "q",
  answer: "Spreads rose 108 bp. [wb-1:3] Yields fell. [arxiv-2503.1:7]",
  abstained: false,
  provider: "mock-extractive",
  top_score: 4.2,
  citations: [
    { doc_id: "wb-1", page: 3 },
    { doc_id: "arxiv-2503.1", page: 7 },
  ],
  invalid_citations: [],
  unsupported_sentences: [],
  passages: [
    { chunk_id: 1, doc_id: "wb-1", page_start: 2, page_end: 3, text: "Spreads rose 108 bp after the reveal.", score: 4.2 },
    { chunk_id: 2, doc_id: "arxiv-2503.1", page_start: 7, page_end: 7, text: "Yields fell.", score: 1.0 },
  ],
};

test("parseAnswer splits text and citations", () => {
  expect(parseAnswer("a [d-1:2] b")).toEqual(["a ", { doc_id: "d-1", page: 2 }, " b"]);
});

test("clicking a citation shows the cited passage and page", () => {
  const onOpenPage = vi.fn();
  render(<AnswerPanel result={result} onOpenPage={onOpenPage} />);
  expect(screen.queryByTestId("cited-passage")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "[wb-1:3]" }));
  const quote = screen.getByTestId("cited-passage");
  expect(quote.textContent).toContain("wb-1, page 3");
  expect(quote.textContent).toContain("Spreads rose 108 bp after the reveal.");
  fireEvent.click(screen.getByRole("button", { name: "Show full page" }));
  expect(onOpenPage).toHaveBeenCalledWith({ doc_id: "wb-1", page: 3 });
});


from langchain_text_splitters import RecursiveCharacterTextSplitter

# sample text to test the splitter on. It contains multiple "sections" separated by double newlines (\n\n)
# and one very long paragraph at the end with no double newlines to test how the splitter handles that edge case
tesla_text = """Tesla's Q3 relults

Tesla reported record revenue of $25.2B Q3 2024

Model Y Performance

The Model Y became the best-sellig vehicle globally, with 350,000 units sold

Production Challenges

Supply Chain caused a 12% increase in production costs

This is one very long paragraph the definitely exceeds our character limit and has no double newlines inside it whatsoever making it impossible to split properly"""

print("\n" + "=" * 60)
print("2. Recursive Character Text Splitetr Solution")
print("=" * 60)

# In characte rtext splitter we can only provide one separator at a time.
# This problem is solved by recursive text splitter
# loop through different separators 
# separator of the most piriority should be at the stat of the list
recursive_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n" , "\n" , ". " , " ", ""],
    chunk_size = 100,
    chunk_overlap = 0
)

# now the splitter will split the text and return a list of chunks 
chunks2 = recursive_splitter.split_text(tesla_text)

 #loop through the chunks and print
for i, chunks in enumerate(chunks2 , 1):
    print(f"Chunk{i}: ({len(chunks)} chars)")
    print(f"{chunks}")
    print()


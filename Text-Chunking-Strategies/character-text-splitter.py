from langchain_text_splitters import CharacterTextSplitter

# sample text to test the splitter on. It contains multiple "sections" separated by double newlines (\n\n)
# and one very long paragraph at the end with no double newlines to test how the splitter handles that edge case
tesla_text = """Tesla's Q3 relults

Tesla reported record revenue of $25.2B Q3 2024

Model Y Performance

The Model Y became the best-sellig vehicle globally, with 350,000 units sold

Production Challenges

Supply Chain caused a 12% increase in production costs

This is one very long paragraph the definitely exceeds our character limit and has no double newlines inside it whatsoever making it impossible to split properly"""

#create an instance of text splitter
#this splitter tries to break text apart wherever it finds the separator
# try different separators
# \n\n , \n and . separators were not able to split the last paragraph so i used " " separator
# " " separotor did split the last pargraph but it chunks were  split making them meaningless
splitter1 = CharacterTextSplitter(
    separator = "\n\n" , # ["\n\n" , "\n" , "." , " " , ""]
    chunk_size = 100,
    chunk_overlap = 0
)

# now call split text to actually run the splitting logic on our paragraph
# it returns a list of string chunks
chunks1 = splitter1.split_text(tesla_text)

# now loopp through each chunk to print it out 
for i, chunk in enumerate(chunks1, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")  #print chunk numver and its character length
    print(f"{chunk}")  # print the chunk content
    print()


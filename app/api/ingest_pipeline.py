import os
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime, timezone
import pandas as pd
import asyncio

from supabase import create_client, Client
from openai import AsyncOpenAI


load_dotenv()
TABLE_NAME = "quran"
# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))


@dataclass
class ProcessedChunk:
    chunk_number: int
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


def get_quran_data(quran_csv_path: str) -> pd.DataFrame:
    # Get Quran Data
    # Read the Quran CSV file
    quran_df = pd.read_excel(quran_csv_path, sheet_name="SuperTable")
    return quran_df


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


async def process_chunk(
    chunk: str,
    chunk_number: int,
    type: str,
    subtype: str = None,
    order_ids=[],
    sura: str = None,
    sura_number: int = None,
    verse_number: int = None,
    footnote_json: dict = None,
    data_json: dict = None,
) -> ProcessedChunk:
    """Process a single chunk of text."""

    # Get embedding
    embedding = await get_embedding(chunk)
    # embedding = [0] * 1536  # FOR TESTING TO DELETE

    if type == "quran":
        # Create metadata for quran
        metadata = {
            "order_ids": order_ids,
            "type": type,
            "subtype": subtype,
            "sura_name": sura,
            "sura_number": sura_number,
            "verse_number": verse_number,
            "footnote_json": footnote_json if footnote_json else None,
            "chunk_size": len(chunk),
            "inserted_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        # Create metadata for none verses
        metadata = {
            "order_ids": order_ids,
            "type": type,
            "data_json": json.dumps(data_json) if data_json else None,
            "chunk_size": len(chunk),
            "inserted_at": datetime.now(timezone.utc).isoformat(),
        }

    return ProcessedChunk(
        chunk_number=chunk_number,
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding,
    )


async def insert_chunk(chunk: ProcessedChunk, total_chunks: int):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "chunk_number": chunk.chunk_number,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
        }

        result = supabase.table(TABLE_NAME).insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None


async def process_insert_quran_document(data: pd.DataFrame, inserted_num_chunks: int = 0):
    """Process Quran document."""
    sura = ""
    counter = 1
    chunk = ""
    type = ""
    subtype = ""
    footnote_json = None
    total_chunks = len(data)
    sura_number = None
    verse_number = None
    order_ids = []

    for index, row in data.iterrows():
        order_id = row["order_id"]
        if row["subtype"] == "sura":
            sura = row["content"]
            chunk = sura + "\n"
            total_chunks -= 1
            order_ids.append(order_id)
            continue
        elif row["subtype"] == "subtitle":
            chunk = row["content"]
            total_chunks -= 1
            order_ids.append(order_id)
            continue
        elif row["subtype"] == "footnote":
            if pd.notna(row["data"]):
                footnote_json = row["data"]
            order_ids.append(order_id)
            content = row["content"]
        elif row["subtype"] == "verse":
            # content = (
            #     lambda r: (
            #         f"{r['main_id']}:{r['minor_id']} {r['content']}"
            #         if r["minor_id"] != "0"
            #         else r["content"]
            #     )
            # )(row)
            content = row["content"]
            order_ids.append(order_id)

        chunk = f"{chunk}\n{content}".strip() if chunk else content

        chunk_number = counter
        type = row["type"]
        subtype = row["subtype"]
        sura_number = row["main_id"]
        verse_number = row["minor_id"] if int(row["minor_id"]) > 0 else None

        processed_chunk = await process_chunk(
            chunk=chunk,
            chunk_number=inserted_num_chunks + chunk_number,
            type=type,
            subtype=subtype,
            sura=sura,
            sura_number=sura_number,
            verse_number=verse_number,
            order_ids=order_ids,
            footnote_json=footnote_json,
        )

        await insert_chunk(processed_chunk, total_chunks)

        counter += 1
        chunk = ""
        content = ""
        footnote_json = None
        sura_number = None
        verse_number = None
        order_ids = []
    if chunk:
        processed_chunk = await process_chunk(
            chunk=chunk,
            chunk_number=inserted_num_chunks + counter,
            type=type,
            subtype=subtype,
            sura=sura,
            sura_number=sura_number,
            verse_number=verse_number,
            order_ids=order_ids,
            footnote_json=footnote_json,
        )
        await insert_chunk(processed_chunk, total_chunks)
    return total_chunks


async def process_insert_glossary_index_document(
    data: pd.DataFrame, start_num_chunks: int = 0
):
    """Process Quran document."""
    counter = 1
    chunk = ""
    type = ""
    total_chunks = len(data)
    order_ids = []

    for index, row in data.iterrows():
        order_id = row["order_id"]
        type = row["type"]
        subtype = row["subtype"]
        if row["subtype"] == "heading1":
            continue
        else:
            if type == "glossary":
                content = row["main_id"] + "\n" + row["content"]
            else:
                content = (
                    row["main_id"]
                    + "\n"
                    + row["minor_id"]
                    + "\n"
                    + (row["sub_id"] + "\n" if pd.notna(row["sub_id"]) else "")
                    + row["content"]
                )
            order_ids.append(order_id)

        chunk = f"{chunk}\n{content}".strip() if chunk else content

        chunk_number = counter

        processed_chunk = await process_chunk(
            chunk=chunk,
            chunk_number=start_num_chunks + chunk_number,
            type=type,
            order_ids=order_ids,
        )

        await insert_chunk(processed_chunk, total_chunks)

        counter += 1
        chunk = ""
        content = ""
        order_ids = []
    if chunk:
        processed_chunk = await process_chunk(
            chunk=chunk,
            chunk_number=start_num_chunks + counter,
            type=type,
            order_ids=order_ids,
        )
        await insert_chunk(processed_chunk, total_chunks)
    return total_chunks


def would_exceed(
    current: str, additional: str, min_chunk_len: int, max_chunk_len: int
) -> bool:
    """
    Determines whether adding 'additional' text to 'current' would exceed the maximum
    allowed chunk length, but only if the current chunk is already at or above the
    minimum chunk length.

    Parameters:
        current (str): The current text in the chunk.
        additional (str): The text we are considering appending.
        min_chunk_len (int): The minimum chunk length desired.
        max_chunk_len (int): The maximum allowed chunk length.

    Returns:
        bool: True if appending the text would cause the chunk to exceed max_chunk_len
              and current chunk is already at least min_chunk_len; otherwise False.
    """
    # Add one extra character (a space) if current is not empty.
    extra = 1 if current else 0
    new_length = len(current) + extra + len(additional)

    # Only flush (i.e. consider it "exceeding") if:
    # 1. The new length is greater than the maximum allowed, AND
    # 2. The current chunk is already sufficiently long (>= min_chunk_len)
    if new_length > max_chunk_len and len(current) >= min_chunk_len:
        return True
    return False


async def process_insert_none_verses(
    df: pd.DataFrame,
    max_chunk_len: int = 1000,
    min_chunk_len: int = 100,
    start_num_chunks: int = 0,
):
    """
    Creates chunks of text from a DataFrame containing headings and paragraphs.

    The DataFrame is assumed to have two columns:
      - "text": The actual text string.
      - "type": Either "heading" or "paragraph".

    Rules:
      - When a heading row is encountered, its text is added to the current chunk.
      - If the next row is also a heading, then its text is appended to the same chunk.
      - When a paragraph row is encountered after one or more headings,
        the paragraph text is appended until the next heading appears.
      - If appending the next row (heading or paragraph) would cause the current chunk
        to exceed max_chunk_len characters, the current chunk is flushed (appended to the chunks list)
        and a new chunk is started with that row's text.
        **Skip "picture"

    Parameters:
        df (pd.DataFrame): DataFrame containing the CSV data.
        max_chunk_len (int): Maximum number of characters allowed per chunk.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    chunks_details = []
    current_chunk = ""
    last_row_was_heading = False
    type = ""
    subtype = ""
    data_json = None
    order_ids = []

    # Iterate over the rows of the DataFrame.
    for _, row in df.iterrows():
        order_id = row["order_id"]
        if pd.isna(row["content"]):
            continue
        text = str(row["content"]).strip()  # ensure text is a string and remove extra spaces
        data_json = row["data"] if pd.notna(row["data"]) else None

        # --- Special processing for subtype ---
        if row["subtype"] == "picture":
            # Skip picture rows.
            continue

        if row["subtype"] in ("table", "list"):
            # For table or list rows we want a separate chunk.
            # If there is a heading in the current chunk, prepend it.
            if last_row_was_heading and current_chunk:
                # Combine the heading and current row's text into a single chunk.
                special_chunk = current_chunk + "\n" + text
                # Clear the current chunk.
                current_chunk = ""
            else:
                type = (
                    f"{row['type']} {int(str(row['main_id'])[-2:])}"
                    if row["type"] == "appendix"
                    else row["type"]
                )
                if current_chunk:
                    chunks.append(current_chunk)
                    chunks_details.append(
                        {"order_ids": order_ids, "type": type, "data_json": data_json}
                    )
                    special_chunk = text
                    order_ids = []
                    current_chunk = ""

            order_ids.append(order_id)
            # Add the special chunk to the chunks list.
            chunks.append(special_chunk)
            # # If there is content in the "data" column, add it to chunk_details.
            # if pd.notna(data_json):
            chunks_details.append(
                {"order_ids": order_ids, "type": type, "data_json": data_json}
            )
            data_json = None

            # After processing a table or list row, treat it as a non-heading.
            last_row_was_heading = False
            order_ids = []
            # Continue to the next row.
            continue

        if row["subtype"].startswith("heading"):
            # If the current chunk already contains paragraph text (i.e. the previous row was not a heading)
            # then we treat this heading as the start of a new chunk.
            if current_chunk and not last_row_was_heading:
                # flush the current chunk into the list
                chunks.append(current_chunk)
                chunks_details.append(
                    {"order_ids": order_ids, "type": type, "data_json": data_json}
                )
                current_chunk = ""
                order_ids = []

            # If adding this heading would exceed the max_chunk_len,
            # flush current_chunk first and then start a new chunk.
            if current_chunk and would_exceed(
                current_chunk, text, min_chunk_len, max_chunk_len
            ):
                chunks.append(current_chunk)
                chunks_details.append(
                    {"order_ids": order_ids, "type": type, "data_json": data_json}
                )
                current_chunk = text
                order_ids = []
            else:
                # If current_chunk is empty, simply start with the heading text.
                # Otherwise, append it (with a space).
                if current_chunk:
                    current_chunk += "\n" + text
                else:
                    current_chunk = text
            type = (
                f"{row['type']} {int(str(row['main_id'])[-2:])}"
                if row["type"] == "appendix"
                else row["type"]
            )
            order_ids.append(order_id)
            last_row_was_heading = True

        else:
            # This row is a paragraph
            # If there is no current chunk (i.e. the text starts with a paragraph) then just start a new one.
            if not current_chunk:
                current_chunk = text
                order_ids.append(order_id)
            else:
                if would_exceed(current_chunk, text, min_chunk_len, max_chunk_len):
                    # If adding this paragraph would exceed the allowed chunk size,
                    # flush the current chunk and start a new one.
                    chunks.append(current_chunk)
                    chunks_details.append(
                        {"order_ids": order_ids, "type": type, "data_json": data_json}
                    )
                    current_chunk = text
                    order_ids = []
                    order_ids.append(order_id)
                else:
                    type = (
                        f"{row['type']} {int(str(row['main_id'])[-2:])}"
                        if row["type"] == "appendix"
                        else row["type"]
                    )
                    current_chunk += "\n" + text
                    order_ids.append(order_id)
            last_row_was_heading = False

    # Flush any remaining text as the final chunk.
    if current_chunk:
        chunks.append(current_chunk)
        chunks_details.append({"order_ids": order_ids, "type": type, "data_json": data_json})

    for i, chunk in enumerate(chunks):
        processed_chunk = await process_chunk(
            chunk=chunk,
            chunk_number=i + 1 + start_num_chunks,
            type=chunks_details[i]["type"],
            order_ids=chunks_details[i]["order_ids"],
            data_json=chunks_details[i]["data_json"],
        )
        await insert_chunk(processed_chunk, len(chunks))

    return len(chunks) + start_num_chunks


async def process_and_store_document(data: pd.DataFrame, type: str, start_num_chunks: int = 0):
    """Process and store all chunks of a document."""
    if type == "quran":
        total_chunks_inserted = await process_insert_quran_document(data)
    elif type == "none_verses":
        total_chunks_inserted = await process_insert_none_verses(
            data,
            max_chunk_len=1000,
            min_chunk_len=100,
            start_num_chunks=start_num_chunks,
        )
    elif type == "glossary_index":
        total_chunks_inserted = await process_insert_glossary_index_document(
            data,
            start_num_chunks=start_num_chunks,
        )
    return total_chunks_inserted


async def main():
    start_time = time.time()

    print("Current working directory:", os.getcwd())
    quran_csv_path = "../../dev/Quran CSV.xlsx"
    quran_df = get_quran_data(quran_csv_path)
    # Assign values where type is 'quran'
    quran_df_verses = quran_df[quran_df["type"] == "quran"]
    quran_df_no_verses = quran_df[~quran_df["type"].isin(["quran", "glossary", "index"])]
    quran_df_glossary_index = quran_df[quran_df["type"].isin(["glossary", "index"])]
    total_chunks = await process_and_store_document(quran_df_verses, "quran")
    total_chunks = await process_and_store_document(
        quran_df_no_verses, "none_verses", start_num_chunks=total_chunks + 1
    )
    total_chunks = await process_and_store_document(
        quran_df_glossary_index, "glossary_index", start_num_chunks=total_chunks + 1
    )

    end_time = time.time()
    print(f"Total time: {((end_time - start_time)/60):.2f} minutes")


if __name__ == "__main__":
    asyncio.run(main())

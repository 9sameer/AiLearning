import docx
import pandas as pd

doc = docx.Document("data.docx")

data = []
for table in doc.tables:
    for row in table.rows:
        row_data = [
            cell.text.strip() if cell.text.strip() != "" else "NULL"
            for cell in row.cells
        ]
        data.append(row_data)

df = pd.DataFrame(data)
print(df)

# ================================================================================================
import pdfplumber
import pandas as pd

rows = []

with pdfplumber.open("data.pdf") as pdf:
    for page in pdf.pages:
        for table in page.extract_tables() or []:
            for row in table:
                # Replace None or empty/whitespace with "NULL"
                clean_row = [(cell.strip() if cell and cell.strip() else "NULL") for cell in row]
                rows.append(clean_row)

# Convert to DataFrame
if rows:
    df = pd.DataFrame(rows)
else:
    df = pd.DataFrame()

print(df.head())

import pandas as pd
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

summary_dtypes = {'Workspace':str, 'Summary':str, 'Top Quotes':str, 'Action Items':str}

df = pd.read_csv('summaries.csv', header=0, dtype=summary_dtypes)
# Replace NaN values with empty strings in the DataFrame
df.fillna('', inplace=True)

# create a new Word document
doc = Document()

# add a heading to the document
doc.add_heading('Partner Center NPS Feedback Summary', 0)

# loop through each row in the DataFrame
for i, row in df.iterrows():
    
    # add a section header
    section_heading = row['Workspace']
    doc.add_heading(section_heading, level=1)
    
    # loop through each column in the row
    for col_name, cell_value in row.items():
        
        # skip the section column
        if col_name == 'Workspace':
            continue
        
        # add a subsection header
        subsection_heading = col_name
        doc.add_heading(subsection_heading, level=2)
        
        # replace '\n' characters with newlines and '[' and ']' with empty strings
        cell_value = cell_value.replace('\\n', '\n').replace('[', '').replace(']', '')

        # add the cell value as a paragraph
        p = doc.add_paragraph(cell_value)
        
        # add some formatting to the paragraph
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.space_after = Inches(0.1)

# save the Word document
doc.save('summary.docx')

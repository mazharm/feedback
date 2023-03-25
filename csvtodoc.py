"""
This module converts a CSV file to a Word document
"""
import pandas as pd
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH  # pylint: disable=no-name-in-module


class CSVtoDOCX:
    """
    This class is used convert a CSV file to a Word document
    """

    def __init__(self):
        return

    def convert(self, dtypes, csv_file, docx_file, heading, sections_column):
        """
        Convert the CSV file to a Word document
        """
        # read the CSV file into a DataFrame
        _df = pd.read_csv(csv_file, header=0, dtype=dtypes)

        # Replace NaN values with empty strings in the DataFrame
        _df.fillna('', inplace=True)

        # create a new Word document
        doc = Document()

        # add a heading to the document
        doc.add_heading(heading, 0)

        # loop through each row in the DataFrame
        for idx, row in _df.iterrows(): # pylint: disable=unused-variable

            # add a section header
            section_heading = row.loc[sections_column]
            doc.add_heading(section_heading, level=1)

            # loop through each column in the row
            for col_name in _df.columns:

                # skip the section column
                if col_name == sections_column:
                    continue

                # add a subsection header
                subsection_heading = col_name
                doc.add_heading(subsection_heading, level=2)

                # replace '\n' characters with newlines and '[' and ']' with empty strings
                cell_value = str(row[col_name]).replace(
                    '\\n', '\n').replace('[', '').replace(']', '')

                # add the cell value as a paragraph
                _p = doc.add_paragraph(cell_value)

                # add some formatting to the paragraph
                _p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                _p.space_after = Inches(0.1)

        # save the Word document
        doc.save(docx_file)
        return



# pylint: disable=pointless-string-statement
"""
This code in place if in case you want to run the tool
as stand alone


summary_dtypes = {'Workspace':str, 'Summary':str, 'Top Quotes':str, 'Action Items':str}

if __name__ == "__main__":
    # create an instance of the CSVtoDOCX class
    csvtodocx = CSVtoDOCX()
    
    # convert the CSV file to a Word document
    csvtodocx.convert(summary_dtypes, 'summaries.csv', 'summary.docx', 
                                'NPS Feedback Summary', 'Workspace')
"""

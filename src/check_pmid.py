import xml.etree.ElementTree as ET
from pathlib import Path
import sys

pmid_to_find = '42316327'  # change this to check different PMIDs

raw_dir = Path('data/raw')
batch_files = sorted(raw_dir.glob('*/batch_*.xml'))

for batch_file in batch_files:
    tree = ET.parse(batch_file)
    root = tree.getroot()
    for article in root.findall('.//PubmedArticle'):
        pmid = article.findtext('.//PMID')
        if pmid != pmid_to_find:
            continue

        title = article.findtext('.//ArticleTitle') or 'No title'
        journal = article.findtext('.//Journal/Title') or 'Unknown'

        print(f'PMID:    {pmid}')
        print(f'Title:   {title}')
        print(f'Journal: {journal}')
        print()

        sections = article.findall('.//Abstract/AbstractText')
        if not sections:
            print('No abstract found.')
        for s in sections:
            label = s.get('Label', 'ABSTRACT')
            text = s.text or ''
            print(f'[{label}]')
            print(text)
            print()
        sys.exit(0)

print(f'PMID {pmid_to_find} not found in any cached batch.')

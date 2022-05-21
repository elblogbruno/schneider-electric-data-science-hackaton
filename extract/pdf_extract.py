import textract
import PyPDF2
import pandas as pd
from tika import parser
import pdfplumber

# PANDAS


# pathfile = "train6/pdfs-1.pdf"
# text = textract.process(pathfile, method='pdfminer')

# pdf_file = open(pathfile, 'rb')
# pdfReader = PyPDF2.PdfFileReader(pdf_file)
# pageObj = pdfReader.getPage(0) 
# print(pageObj.extractText())
# pageObj.close()

raw = parser.from_file("train6/pdfs-1.pdf")
# print(raw['content'])
to_process = raw['content'].split('\n')

te = {  'test_index': '', 
        'countryName': '', 
        'EPRTRSectorCode': '', 
        'eprtrSectorName': '', 
        'EPRTRAnnexIMainActivityCode': '',
        }


lines = []
for line in to_process:
    lines.append(line)
    


data_cleaned = [x for x in lines if x != '']
data_splited = [x.split(':') for x in data_cleaned]


# for x in data_splited:
#     print(x)



# for x in data_splited:
#     print(x)
        
        
# dic = {}
# for dat in data_splited:
#     dic["dat"] = 


    
# pd_dict = pd.DataFrame(te)
# print(pd_dict)
    

with pdfplumber.open("train6/pdfs-1.pdf") as pdf: 
    text = pdf.pages[0]
    keys = text.filter(lambda obj: not (obj["object_type"] == "char" and "Bold" not in obj["fontname"]))
    values = text.filter(lambda obj: not (obj["object_type"] == "char" and "Bold" in obj["fontname"]))
    key_ext = keys.extract_text()
    val_ext = values.extract_text()
    
    dicto = {}
    
    
    keys_splitted = key_ext.split("\n")
    values_splitted = val_ext.split("\n")
    
    
    names = []
    result = []
    
     
    for n in keys_splitted:
        splitted = n.split(":")
        clean = list(filter(None, splitted))
        
        for c in clean:
            j = c.replace(' ', '')
            names.append(j)
        

    for i in range(0, len(values_splitted)):
        print(values_splitted[i])

            
        
        
        
        
    # for x in names:
    #     print(x)
    # print(len(names))
    # print(len(values_splitted))
    
    # for i in range(len(names)):
    #     dicto[names[i]] = values_splitted[i]
        
    
    # print(dicto)
        
    # print(names)        
        # for val in splitted:
        #     val = val.strip("\n")
        #     for k in val.split(":"):
        #         panda[k] = val
              
              
                
    # for p in panda:
    #     print(p)
        
        
    
   
   
   
    
    











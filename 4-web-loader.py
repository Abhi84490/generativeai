# from langchain_community.document_loaders import WebBaseLoader


# url="https://www.flipkart.com/apple-macbook-air-m2-8-gb-256-gb-ssd-mac-os-monterey-mly33hn-a/p/itmdc5308fa78421?pid=COMGFB2GMCRXZG85&lid=LSTCOMGFB2GMCRXZG855GPGWQ&marketplace=FLIPKART&q=mackbook&store=6bo%2Fb5g&srno=s_1_1&otracker=search&otracker1=search&fm=organic&iid=en_s8pZxCGenkBT_H0pRHP2hPMft3frVC8k6eCNtIicrwn_OmVOUQbdoPMsgjECK0NI_zcMCBd4e2NxAaDxXs2U9_UFjCTyOHoHZs-Z5_PS_w0%3D&ppt=hp&ppn=homepage&ssid=be0gchyy4g0000001757684444921&qH=c0a42b6ebf7777b3"
# loader=WebBaseLoader(url)

# docs =loader.load()
# print(len(docs))
# print(docs[0].page_content)
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
load_dotenv()
import os


model = ChatOpenAI(
    base_url="https://models.github.ai/inference",  # GitHub Models endpoint
    api_key=os.environ["GITHUB_TOKEN"],             # PAT from .env
    model="gpt-4o-mini",                            # or "gpt-4o", "gpt-3.5-turbo"
    temperature=0.7
)

prompt=PromptTemplate(
    template='Answer the following question \n {question} from the following text -\n {text}',
    input_variables=['question','text']
)

parser=StrOutputParser()



# Fake browser header
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0.0.0 Safari/537.36"}

url = "https://www.flipkart.com/apple-macbook-air-m2-8-gb-256-gb-ssd-mac-os-monterey-mly33hn-a/p/itmdc5308fa78421"

loader = WebBaseLoader(url, header_template=headers)
docs = loader.load()

# print(len(docs))
# print(docs[0].page_content[:1000])  # print only first 1000 chars
chain= prompt | model | parser
result=chain.invoke({"question":"What the max sound of this product? ", "text":docs[0].page_content})
print(result)
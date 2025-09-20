import json
import os

import requests
from dotenv import load_dotenv

from helper import get_data_from_serpapi, get_serpapi_url
load_dotenv()

def fetch_patent_data(query, dir_path):
    """
    Fetch patent data from SerpAPI and save it to the specified directory.
    """
    api_key=os.getenv("SerpApi_API_KEY")
    if not api_key:
        raise ValueError("SerpApi_API_KEY environment variable not set")
    
    os.makedirs(dir_path, exist_ok=True)
    url=(
        f"https://serpapi.com/search?engine=google_patents&q={query}&api_key={api_key}"
    )

    response=requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching data:{response.status_code},{response.text}")
        exit(1)


    if response.status_code == 200:
        data = response.json()
        for idx, patent in enumerate(data.get("organic_results", [])):
            serpapi_url = get_serpapi_url(patent)
            response_data=get_data_from_serpapi(serpapi_url)
            if not response_data:
                print(f"Error fetching data for patent {idx}: No data returned")        
                continue

            with open(f"{dir_path}/patent_data{idx+1}.json", "w") as f:
                json.dump(response_data, f, indent=2)


            patent_citations=response_data.get("patent_citations",{}).get(
                "original",{}
            )


            for idx2, citation in enumerate(patent_citations):
                serpapi_url2=citation.get("serpapi_url",None)
                if serpapi_url2:
                    citation_data=get_data_from_serpapi(serpapi_url2)
                    if citation_data:
                        with open(f"{dir_path}/citation_{idx}_{idx2}.json","w")as f:
                            json.dump(citation_data, f, indent=2)
                    
                    else:
                        print(
                            f"Error fetching data for citation {idx2}: No data returned."
                        )       
                else:
                    print(f"No SERPAPI link found for citation {idx2}.")

    else:
        print(f"Error:{response.status_code},{response.text}")


if __name__ == "__main__":
    query=input("Enter the search query for patents:")
    # dir_path=input("Enter the directory path to save results:")
    dir_path="results/patents"
    try:
        fetch_patent_data(query,dir_path)
        print(f"patent data fetched and saved to '{dir_path}'")
    except Exception as e:
        print(f"Error occurred: {e}")
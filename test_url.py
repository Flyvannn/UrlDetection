import requests
import urllib3
import pandas as pd
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

def is_access(url):
    try:
        if not url.startswith("http"):
            url = "http://" + url
        response = requests.get(url, headers=headers, verify=False, allow_redirects=True, timeout=0.5)
        # print(response.status_code)
        if response.status_code == 200:
            return True
        else:
            raise requests.RequestException(u"Status code error: {}".format(response.status_code))
    except requests.RequestException as e:
        return False

if __name__ == '__main__':
    df = pd.read_csv("data/new_phishingUrls.csv")
    print(len(df))
    # df2 = df[df['label']==0]
    # df2 = df2.sample(100000)
    # print(len(df2))
    df['access'] = df['url'].apply(is_access)
    df3 = df[df['access'] == True]
    print(len(df3))
    df3.to_csv("data/new_accessPhishing.csv", index=False)

    # url = "https://sandy.cesy.top/"
    # print(is_access(url))


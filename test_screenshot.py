from selenium import webdriver
import os,time

url = "https://zhuanlan.zhihu.com/p/476650258"
file_name = url.replace("/", "_").replace(":",  "_")

# driver =webdriver.Firefox()
options =webdriver.FirefoxOptions()
options.add_argument("--headless") #设置火狐为headless无界面模式
options.add_argument("--disable-gpu")
options.add_argument("--disable-gpu")
driver = webdriver.Firefox(options=options)


driver.get(url)

pic_path = './image/'+ file_name +'.png'

driver.save_screenshot(pic_path)






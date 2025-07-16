import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re

from selenium.webdriver.support.wait import WebDriverWait

from globalposition import *


class GoogleMapsScraper:
    def __init__(self, json_file_path, c_value=13,threshold = 4):
        """
        初始化爬虫
        :param json_file_path: JSON文件路径
        :param c_value: 常数C，用于计算奖励值
        """
        self.json_file_path = json_file_path
        self.c_value = c_value
        self.driver = None
        self.threshold = threshold

    def setup_driver(self):
        """设置Chrome驱动"""
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        # 如果需要无头模式，取消下面的注释
        # options.add_argument('--headless')

        self.driver = webdriver.Chrome(options=options)

    def search_place(self, place_name):
        """
        在Google Maps搜索地点
        :param place_name: 地点名称
        :return: 搜索结果URL
        """
        search_url = f"https://www.google.com/maps/search/{place_name}+singapore"
        print(f"{search_url} 正在搜索...")
        self.driver.get(search_url)
        time.sleep(3)  # 等待页面加载

    def get_rating_and_reviews(self, place_name):
        """
        获取地点的评分和评论数
        :param place_name: 地点名称
        :return: (rating, review_count) 或 (None, None)
        """
        try:
            self.search_place(place_name)

            # 等待评分元素出现
            rating = None
            review_count = None
            spans = self.driver.find_elements(By.CSS_SELECTOR, 'span[aria-label]')
            pattern1 = r"([0-9.]+)\s+stars\s+(\d+)\s+Reviews"
            pattern2 = r'([0-9.]+)\s+星级\s+([\d,]+)条评价'
            for span in spans:
                aria_label = span.get_attribute('aria-label')
                # print(f"检查aria-label: {aria_label}")
                match1 = re.search(pattern1, aria_label)
                match2 = re.search(pattern2, aria_label)
                if match1:
                    rating = float(match1.group(1))
                    review_count = int(match1.group(2).replace(',', ''))
                    print(f"找到评分: {rating}, 评论数: {review_count} for {place_name}")
                    return rating, review_count
                elif match2:
                    rating = float(match2.group(1))
                    review_count = int(match2.group(2).replace(',', ''))
                    print(f"找到评分: {rating}, 评论数: {review_count} for {place_name}")
                    return rating, review_count
            return None, None
        except TimeoutException:
            print(f"加载 {place_name} 时超时，请检查网络连接或地点名称是否正确。")
            return None, None

    def calculate_reward(self, rating, review_count):
        """
        计算奖励值
        :param rating: 评分
        :param review_count: 评论数
        :return: 奖励值
        """
        if rating is None or review_count is None:
            return 0
        reward = (rating - self.threshold) * review_count / self.c_value
        if reward > self.threshold:
            reward = self.threshold
        elif reward < -self.threshold:
            reward = -self.threshold
        return reward

    def process_json_file(self):
        """处理JSON文件，更新奖励值和添加新字段"""
        # 读取JSON文件
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.setup_driver()

        try:
            # 处理每个地点
            reward_cache = {}
            for i, location in enumerate(data):
                place_name = location['name']
                print(f"正在处理 ({i + 1}/{len(data)}): {place_name}")

                # 获取评分和评论数

                rating, review_count = self.get_rating_and_reviews(place_name)

                new_reward = self.calculate_reward(rating, review_count)
                location['reward'] = round(new_reward, 2)
                location['rating'] = rating
                location['review_count'] = review_count
                reward_cache[place_name] = {
                    'reward': location['reward'],
                    'rating': rating,
                    'review_count': review_count
                }
                print(f"  评分: {rating}, 评论数: {review_count}, 新奖励值: {new_reward:.2f}")

                # 添加延迟避免被封
                time.sleep(2)

            for i, location in enumerate(data):
                # 创建neighbors_reward
                neighbors_reward = {}
                for neighbor, distance in location['neighbors'].items():
                    if neighbor in reward_cache:
                        neighbor_info = reward_cache[neighbor]
                        neighbor_reward = neighbor_info['reward']
                        reward_cache[neighbor] = {
                            'reward': neighbor_reward,
                            'rating': neighbor_info['rating'],
                            'review_count': neighbor_info['review_count'],
                            'adjusted_value': round(distance - neighbor_reward, 2)
                        }
                    else:
                        n_rating, n_review_count = self.get_rating_and_reviews(neighbor)
                        time.sleep(2)  # 添加延迟
                        if n_rating is not None and n_review_count is not None:
                            neighbor_reward = round(self.calculate_reward(n_rating, n_review_count), 2)
                        else:
                            neighbor_reward = 0
                        reward_cache[neighbor] = {
                            'reward': neighbor_reward,
                            'rating': n_rating,
                            'review_count': n_review_count,
                            'adjusted_value': round(distance - neighbor_reward, 2)
                        }
                    neighbors_reward[neighbor] = reward_cache[neighbor]

                location['neighbors_reward'] = neighbors_reward

        finally:
            if self.driver:
                self.driver.quit()

        # 保存更新后的JSON文件
        output_file = self.json_file_path.replace('.json', '_updated.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n处理完成！更新后的文件保存为: {output_file}")

        return data

    def get_specific_review(self, place_name):
        """
        Extracts all reviews for a specific place from Google Maps.
        This function will first search for the place, then click on the 'Reviews' tab,
        scroll through all the reviews, and extract the text.

        :param place_name: The name of the place to get reviews for.
        :return: A list of all review texts.
        """
        self.setup_driver()
        try:
            self.search_place(place_name)
            wait = WebDriverWait(self.driver, 10)

            # Find and click the button to go to the reviews page/tab.
            reviews_button_xpath = "//button[contains(@aria-label, 'Reviews') or .//*[text()='Reviews'] or .//*[text()='评价']]"
            reviews_button = wait.until(EC.element_to_be_clickable((By.XPATH, reviews_button_xpath)))
            reviews_button.click()
            time.sleep(3)

            # Find the scrollable element containing reviews.
            try:
                scrollable_div = wait.until(EC.presence_of_element_located((By.XPATH, '//div[@class="wiI7pd"]')))
            except TimeoutException:
                print("Could not find dedicated scrollable review feed. Scrolling the main page.")
                scrollable_div = self.driver.find_element(By.XPATH, "/html/body")

                # Scroll down to load all reviews
            last_height = self.driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
            while True:
                self.driver.execute_script("arguments[0].scrollTo(0, arguments[0].scrollHeight);", scrollable_div)
                time.sleep(2)
                new_height = self.driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
                if new_height == last_height:
                    break
                last_height = new_height

                # Find all review text elements based on the provided span class.
            review_elements = self.driver.find_elements(By.CSS_SELECTOR, "span.wiI7pd")
            reviews = [elem.text for elem in review_elements if elem.text and elem.is_displayed()]

            print(f"Successfully extracted {len(reviews)} reviews for {place_name}.")
            return reviews

        except (TimeoutException, NoSuchElementException) as e:
            print(f"An error occurred while fetching reviews for {place_name}: {e}")
            return []
        finally:
            if self.driver:
                self.driver.quit()


            # 使用示例
# if __name__ == "__main__":
#     # 设置参数
#     json_file = "Graph/GUIDE_037.json"  # 您的JSON文件路径
#
#     get_position(json_file, json_file)
#     c_constant = 500  # 可以根据需要调整这个值
#
#     # 创建爬虫实例
#     scraper = GoogleMapsScraper(json_file, c_constant,threshold=4)
#
#     # 处理文件
#     updated_data = scraper.process_json_file()
#
#     # 打印一些统计信息
#     total_locations = len(updated_data)
#     successful_updates = sum(1 for loc in updated_data if loc.get('rating') is not None)
#     print(f"\n统计信息:")
#     print(f"总地点数: {total_locations}")
#     print(f"成功更新: {successful_updates}")
#     print(f"更新率: {successful_updates / total_locations * 100:.1f}%")
if __name__ == "__main__":
    json_file = "TEST_graph.json"
    c_constant = 500

    # Assuming get_position is defined in the imported calculator module
    get_position(json_file, json_file)

    scraper = GoogleMapsScraper(json_file, c_constant, threshold=4)
    updated_data = scraper.process_json_file()

    total_locations = len(updated_data)
    successful_updates = sum(1 for loc in updated_data if loc.get('rating') is not None)
    print(f"\n统计信息:")
    print(f"总地点数: {total_locations}")
    print(f"成功更新: {successful_updates}")
    print(f"更新率: {successful_updates / total_locations * 100:.1f}%")

    # --- Demonstration of the new get_specific_review function ---
    print("\n--- Testing Review Extraction Function ---")
    if updated_data:
        # Using the first location from the data as an example
        sample_place_name = updated_data[0]['name']

        # Create a new instance or reuse the old one to call the function
        review_scraper = GoogleMapsScraper(json_file, c_constant, threshold=4)
        all_reviews = review_scraper.get_specific_review(sample_place_name)

        print(len(all_reviews))

        print(all_reviews)

        if all_reviews:
            print(f"\n--- Sample Reviews for {sample_place_name} ---")
            # Print the first 5 reviews as a sample
            for i, review in enumerate(all_reviews[:5]):
                print(f"Review {i + 1}:\n{review}\n")
        else:
            print(f"No reviews were found for {sample_place_name}.")

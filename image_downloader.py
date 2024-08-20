import requests
import threading
import uuid


folder = None


def download_image(query, url, index):
    try:
        with requests.get(url, timeout=10) as r:
            filename = f"{query}"
            with open(f"{folder}/{filename}.jpg", "wb") as f:
                f.write(r.content)

        print(f"image{index} downloaded......")
    except:
        pass


def initialize_threads(query, urls):
    threads = []
    index = 1
    for url in urls:
        t = threading.Thread(target=download_image, args=(query, url, index))
        index += 1
        threads.append(t)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def download(query, urls):
    initialize_threads(query, urls)
import base64
import json
import requests


def detect_text(path):
    with open(path, 'rb') as image_file:
        content = base64.b64encode(image_file.read())
        content = content.decode('utf-8')

    api_key = "AIzaSyAJQbMCNG-hZ4dXoXWz2Ho9clIIySvHseM"
    url = "https://vision.googleapis.com/v1/images:annotate?key=" + api_key
    headers = {'Content-Type': 'application/json'}
    request_body = {
        'requests': [
            {
                'image': {
                    'content': content
                },
                'features': [
                    {
                        'type': "TEXT_DETECTION",
                        'maxResults': 10
                    }
                ]
            }
        ]
    }
    response = requests.post(
        url,
        json.dumps(request_body),
        headers
    )

    result = response.json()
    if result != {'responses': [{}]}:
        print("result", result)
        return result['responses'][0]['textAnnotations'][0]['description']
    # print(result['responses'][0]['textAnnotations'][0]['description'])
    # print(result['responses'])


if __name__ == '__main__':
    txtName = "./22.txt"
    f = open(txtName, "a+")
    text = detect_text("./static/images/testImage/WeChat Screenshot_20190718150025.png")
    print("text", text)
    text = str(text)
    f.write(text)
    f.close()

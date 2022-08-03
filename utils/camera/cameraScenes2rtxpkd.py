import sys
import json

if __name__ == "__main__":
    cameraScenesFile = open(sys.argv[1], "r")
    outfile = open(sys.argv[2], "w")
    cameras = json.load(cameraScenesFile)
    counter = 0
    json_string = '['
    for item in cameras:
        if counter == 0:
            json_string = json_string + '{{"cmd":"--camera {} {} {} {} {} {} {} {} {}"}}'.format(item["position"][0], item["position"][1], item["position"][2],
                item["position"][0] + item["direction"][0], item["position"][1] + item["direction"][1], item["position"][2] + item["direction"][2],
                item["up"][0], item["up"][1], item["up"][2])
            counter = 1
        else:
            json_string = json_string + ',{{"cmd":"--camera {} {} {} {} {} {} {} {} {}"}}'.format(item["position"][0], item["position"][1], item["position"][2],
                item["position"][0] + item["direction"][0], item["position"][1] + item["direction"][1], item["position"][2] + item["direction"][2],
                item["up"][0], item["up"][1], item["up"][2])
    json_string = json_string + "]"
    outfile.write(json_string)
    outfile.close()
    cameraScenesFile.close()

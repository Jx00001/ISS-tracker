import requests
import plotly.express as px
import plotly.graph_objects as go
import plotly as plt
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import cv2

cv2.useOptimized()

API_url = "http://api.open-notify.org/iss-now.json"
overlay_image = cv2.imread('iss_pointer.png', cv2.IMREAD_UNCHANGED) #import satellite png image
fx = cv2.FONT_HERSHEY_SIMPLEX

globals()["Storeddata"] = {'Object/time': [], 'GPSLat': [], 'GPSLng': []} #dict DataFrame
globals()["Zoom"] = 2

def on_zoom(val):
    globals()["Zoom"] = val

def PlotGPSdata(longt, lat, obj, time):
    """ Ploting function takes the scrapped data (time, Lat, Lng) and plots it into a map """
    globals()["Storeddata"]["Object/time"].append(f"ISS-{time}")
    globals()["Storeddata"]["GPSLat"].append(float(lat))
    globals()["Storeddata"]["GPSLng"].append(float(longt))

    df = pd.DataFrame(globals()["Storeddata"])

    fig = px.scatter_mapbox(df,
                            lat="GPSLat", lon="GPSLng",
                            hover_name='Object/time', zoom=globals()["Zoom"],
                            height=600, width=1100,
                            mapbox_style='open-street-map',
                            center=go.layout.mapbox.Center(
                                lat=float(lat),lon=float(longt)
                                )
                            )

    img = plt.io.to_image(fig, format='png')

    # converting the map bytesData into an Opencv image dataType
    buf = BytesIO(img)
    img = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    return img

def add_transparent_image(background, foreground, x_offset=None, y_offset=None): 
    """ a weird function i scrapped online, it really just for overlaying a trensparent picture above a background."""
    """ it really does run faster than any code i tried. so enjot it :) """
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2
    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)
    if w < 1 or h < 1: return
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

while True:
    API_call = requests.get(API_url)
    data = API_call.json() # request the position of the ISS from the API

    position, UnixTime = data["iss_position"], data["timestamp"]
    lg, lt =  position["longitude"], position["latitude"] # Scrap UnixTime, Position.

    plot = PlotGPSdata(lg, lt, "ISS", UnixTime) # Plotting the scrapped data into a map.
    img = plot

    # the weird function in action
    add_transparent_image(img, cv2.resize(overlay_image, (60, 60), interpolation=cv2.INTER_AREA), 518, 258)

    cv2.putText(img, f"time : {UnixTime}", (92, 110), fx, 1, (0, 0, 0), 1)
    cv2.putText(img, f"longitude : {lg}", (92, 150), fx, 1, (0, 0, 0), 1)
    cv2.putText(img, f"latitude : {lt}", (92, 190), fx, 1, (0, 0, 0), 1)

    cv2.imshow('ISS Current position', img)
    if len(globals()["Storeddata"]["Object/time"]) == 1:
        cv2.createTrackbar('zoom', 'ISS Current position', 1, 5, on_zoom)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"ISS_data/ISS-{UnixTime}.png", img)


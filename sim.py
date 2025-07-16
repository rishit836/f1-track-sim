from track import create_track

tracks = create_track("all", False)
for keys in tracks.keys():
    print(keys,"track generated and stored.")
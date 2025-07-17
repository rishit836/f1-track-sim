from track import create_track
import os
import pickle


class racing_car():
    def __init__(self):
        pass


class sim_environment():
    def __init__(self,verbose:bool=True,tracks_all_or_not:bool=True):
        self.verbose = verbose
        self.track_list = list(self.get_track_list().keys())
        self.current_track = self.track_list[0]
        self.car_angle = self.calculate_angle()

    def calculate_angle(self):
        pass
    def get_track_list(self):
        if os.path.exists("track_data.pkl"):
            with open("track_data.pkl","rb") as f:
                track_list = pickle.load(f)
        else:
            track_list = create_track(verbose=False)
            with open('track_data.pkl',"wb") as f:
                if self.verbose:
                    print("saved the file as file didnt exist.")
                pickle.dump(track_list,f)
        return track_list
    

    def reset(self):
        self.track_list = list(self.get_track_list().keys())
        self.current_track = self.track_list[0]
        # using the first value in dictionary for the track as a starting position
        self.x,self.y = self.track_list[self.current_track][0][0],self.track_list[self.current_track][0][1]
        self.start_pos = (self.x,self.y)


    def get_start_pos(self):
        for track in self.track_list:
            print(track)

        
if __name__ == "__main__":
    s = sim_environment()
    track_list = s.get_track_list()
    with open("track.txt","w") as f:
        f.write(str(track_list))
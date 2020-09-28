import animator.triangulation as triangulation
from animator.animation import *

class Animation:
    def __init__(self, frames):
        assert len(frames) != 0
        self.frames = frames
        self.ptr = 0

    def frame(self):
        return self.frames[self.ptr]

    def reset(self):
        self.ptr = 0

    def update(self):
        if self.ptr == len(self.frames)-1:
            self.ptr = 0
        else:
            self.ptr += 1

class Animator:
    def __init__(self, drawing, bones):
        self.drawing = drawing
        self.ratio = 1.0
        self.bones = bones

        # Skinning(triangulation and weight calculation)
        t_start = time.time()
        img_gray = cv2.cvtColor(self.drawing, cv2.COLOR_BGR2GRAY)
        contour = triangulation.contour(img_gray)
        keypoints = triangulation.keypoints_uniform(img_gray, contour)
        triangles_unconstrained, edges = triangulation.triangulate(contour, keypoints)
        t_tri = time.time()-t_start
        self.triangles = triangulation.constrain(contour, triangles_unconstrained, edges)
        t_constrain = time.time()-t_start-t_tri
        self.weights = calcWeights(self.bones, self.triangles)
        t_weights = time.time()-t_start-t_constrain

        # print('Triangulation', t_tri)
        # print('Constrained Triangulation', t_constrain)
        # print('Weights', t_weights)

        self.current_frame = None

    def update(self):
        pass

    def generate_animation(self, bones_frames):
        frames = []

        for i in range(len(bones_frames)):
            # Find same frame
            index_same = -1
            for j in range(i):
                if (bones_frames[i] == bones_frames[j]).all():
                    index_same = j

            if index_same != -1:
                # Copy frame
                frames.append(frames[index_same])
            else:
                bones_n = bones_frames[i]
                triangles_next = animate(self.bones,bones_n,self.triangles,self.weights)
                img_n, anchor, mask_img_n = warp(self.drawing, self.triangles, triangles_next, bones_n[0])


                img_n = cv2.resize(img_n, None, fx=self.ratio, fy=self.ratio)
                anchor = anchor*self.ratio
                mask_img_n = cv2.resize(mask_img_n, None, fx=self.ratio, fy=self.ratio)

                frames.append((img_n, anchor, mask_img_n))

        return Animation(frames)

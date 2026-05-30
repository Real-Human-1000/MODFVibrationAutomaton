import numpy as np
import math
import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi'] = 600
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from MultiSystem import MultiSystem
from random import randint
import wavio
# Render the provided systems on a 2d canvas
# This script started with good intentions but has become a complete mess
# I really should make videos in actual animation software
# Or like a game engine or smth smh my head

def pyplot_animate_2_systems(sysX:MultiSystem, sysY:MultiSystem, t_start, t_stop):
    output_path = os.path.join("output", "pyplot_result.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps = 60
    frame_width = 480
    frame_height = 480
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_width, frame_height))

    fig, ax = plt.subplots()

    t = np.linspace(t_start, t_stop, num=1800)
    tx, xx = sysX.get_response(t_start, t_stop, t)
    ty, xy = sysY.get_response(t_start, t_stop, t)

    # Draw animated system
    sys_pack = [(sysX, 0, tx, xx), (sysY, np.pi/2, ty, xy)]  # system and its axis
    for tidx in range(len(t)):
        # Loop for every time step
        print(f"Rendering frame {tidx} (t={t[tidx]:.02f}); {100*tidx/len(t):.01f}% done")
        ax.cla()
        for s in range(2):
            # For each of our systems
            sys, angle, t, y = sys_pack[s]
            rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            sep = 3  # default spring length / separation between masses
            low_lim = -sep * ((sys.n - 1) / 2 + 1)
            up_lim = sep * ((sys.n - 1) / 2 + 1)
            plt.xlim(low_lim, up_lim)
            plt.ylim(low_lim, up_lim)
            for i in range(sys.n):
                # For each mass/spring
                mass_point = rot_mat @ np.array([[low_lim + sep * (i + 1) + y[i, tidx], 0]]).T
                ax.plot([mass_point[0, 0]], [mass_point[1, 0]], 'ro', markersize=5 * np.sqrt(sys.M[i, i]), zorder=10)
                if i == 0:
                    # Spring connects to wall
                    wall_point = rot_mat @ np.array([[low_lim, 0]]).T
                    ax.plot([wall_point[0,0], mass_point[0,0]], [wall_point[1,0], mass_point[1,0]], 'r', linewidth=(5 * sep / abs(sep + y[i,tidx])), zorder=1, solid_capstyle='butt')
                else:
                    # Connect to the previous mass
                    prev_point = rot_mat @ np.array([[low_lim + sep * (i + 1) - sep + y[i-1,tidx], 0]]).T
                    ax.plot([prev_point[0,0], mass_point[0,0]], [prev_point[1,0], mass_point[1,0]], 'g', linewidth=(5 * sep / abs(sep + y[i,tidx] - y[i-1,tidx])), zorder=1, solid_capstyle='butt')
                if i == sys.n-1:
                    # Connect to the other wall
                    wall_point = rot_mat @ np.array([[up_lim, 0]]).T
                    ax.plot([mass_point[0,0], wall_point[0,0]], [mass_point[1,0], wall_point[1,0]], 'b', linewidth=(5 * sep / abs(sep - y[i,tidx])), zorder=1, solid_capstyle='butt')

            # Draw rails
            upper_rail_points = rot_mat @ (y[3,tidx] * np.array([[1, 1], [0, 0]]) + np.array([[0.025, 0.025], [-1, 1]]))
            plt.plot([upper_rail_points[0,0], upper_rail_points[0,1]], [upper_rail_points[1,0], upper_rail_points[1,1]], 'k')
            lower_rail_points = rot_mat @ (y[3, tidx] * np.array([[1, 1], [0, 0]]) + np.array([[-0.025, -0.025], [-1, 1]]))
            plt.plot([lower_rail_points[0, 0], lower_rail_points[0, 1]], [lower_rail_points[1, 0], lower_rail_points[1, 1]], 'k')

        # Draw sketch trajectory
        plt.plot(xx[3,:tidx], xy[3,:tidx], 'k')

        fig.canvas.draw()
        graph_img = Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.tostring_argb())
        # ^ The RGBA channels are messed up, but we just need to be aware of that
        graph_arr = np.array(graph_img)
        graph_arr = np.stack((graph_arr[:, :, 1], graph_arr[:, :, 2], graph_arr[:, :, 3]), axis=2)  # 0 is A
        graph_img = Image.fromarray(graph_arr)
        graph_img = graph_img.resize((frame_width, frame_height))

        out.write(np.array(graph_img))

    out.release()
    cv2.destroyAllWindows()

def int_tuple(tup):
    # Convert all elements in a tuple to an int
    return tuple([int(t) for t in tup])

def match_phase(f0, p0, f1, t):
    # Find the new phase required to allow the wave to transition from frequency f0 (with phase p0) to frequency f1 seamlessly
    return p0 + 2 * math.pi * t * (f0 - f1)

def string_freq(L, T, mu, n=1):
    # Calculate the nth resonant frequency of a stretched string
    # n is the number of the harmonic (1 for fundamental)
    # L is the length of the string in m (or whatever)
    # T is the tension that the string is under, in N (or whatever)
    # mu is the string's linear density, in kg / m (or whatever)
    return (n / (2 * L)) * math.sqrt(T / mu)

def triangle_wave(n, f, p, t):
    # Approximate a triangle wave with n sine waves
    return (-8 / (math.pi**2)) * sum([
            ((-1)**k) / ((2 * k - 1)**2) *
            math.sin(
                2 * math.pi * (2 * k - 1) * (f * t + p / (2 * math.pi))
            )
        for k in range(n)])

def guitar_wave_amp(k):
    # Amplitude of waves for the additive synthesis of a plucked string sort of sound
    return 0.75 * math.exp(-((k-2)/0.5)**2) + 0.5 * math.exp(-((k-5)/1)**2) + 0.15 * math.exp(-((k-12)/6)**2)

def guitar_string_wave(n, f, p, t):
    # Following the chart from https://www.acs.psu.edu/drussell/demos/string/fixed.html
    # This might be approximating the shape of the guitar string and not the audio waveform :|
    # Their chart can be fit to the equation y = 1 - 0.28975 * ln(x) (without harmonics that are multiples of 3)
    # This only goes up to n=31 (range(32)) before the ln function turns negative
    # 1.02433 * math.exp(-0.112381 * k) is similar but approaches 0 asymptotically
    #return sum([(1 - 0.28975 * math.log(k)) * math.sin(2 * math.pi * k * f * t + k * p) for k in range(min(n,31)) if k % 3 != 0])
    return sum([(1 - (1/20000) * k * f) * guitar_wave_amp(k) * math.sin(2 * math.pi * k * f * t + k * p) for k in range(n) if k % 3 != 0])

def draw_mass(draw, pt, rad):
    # LightSteelBlue: (176, 196, 222)
    # SlateGray: (112, 128, 144)
    # LightSlateGray: (119, 136, 153)
    # AliceBlue: (240, 248, 255)
    draw.circle(pt, radius=rad, fill=(119, 136, 153+20))  # Reflection light
    draw.circle((pt[0], pt[1]-(rad-4*rad/5)), radius=4*rad/5, fill=(112, 128, 144))  # Base
    #draw.ellipse([(pt[0]-2*rad/5, pt[1]-8*rad/9), (pt[0]+6*rad/7, pt[1]+rad/3)], fill=(222-50, 196-50, 176-50))  # Highlight
    # For the highlight I want to draw a sort of clipped circle
    # But I don't want to deal with layering issues or masks
    points = []
    for i in range(int(2*math.pi*5)):  # This probably isn't very efficient
        theta = i/5
        newpt = (pt[0] + rad/2 + rad * math.cos(theta), pt[1] - rad/2 - rad * math.sin(theta))
        if ((newpt[0] - pt[0])**2 + (newpt[1] - pt[1])**2) > rad**2:
            # Clip to edge of main circle
            vec = (newpt[0] - pt[0], newpt[1] - pt[1])
            mag = (vec[0]**2 + vec[1]**2)**0.5
            newpt = (rad * vec[0] / mag + pt[0], rad * vec[1] / mag + pt[1])
        points.append(newpt)
    draw.polygon(points, fill=(176-50, 196-50, 222-50))  # Highlight
    #draw.circle((pt[0]+rad/2, pt[1]-rad/2), radius=rad/6, fill=(255-20, 248-20, 240-20))  # Specular
    draw.ellipse([(pt[0] + 2*rad/5 - rad/8, pt[1] - rad/2 - rad/10), (pt[0] + 2*rad/5 + rad/8, pt[1] - rad/2 + rad/10)],
                 fill=(240-20, 248-20, 255-20))  # Specular
    # ^ could change this to move the specular splotch relative to an invisible light source ig

def k_to_coils(k):
    # Low k has many coils
    # High k has few coils
    return 6 + int(5000 / (k + 300))

def draw_spring(draw, start_pt, end_pt, spring_width, stroke_width, n_coils=5):
    # All values in pixels of the draw context
    # start_pt and end_pt should be 2-tuples
    # Springs have two straight ends and a wavy middle, like this: -WWW-
    end_len = 0.25
    dir_vec = (end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
    mag = ((dir_vec[0])**2 + (dir_vec[1])**2)**0.5
    dir_vec = (0 if mag==0 else dir_vec[0] / mag, 0 if mag==0 else dir_vec[1] / mag)  # normalized
    orth_vec = (-dir_vec[1], dir_vec[0])  # normalized

    next_pt = (start_pt[0] + end_len * mag * dir_vec[0], start_pt[1] + end_len * mag * dir_vec[1])
    # Draw middle
    middle_segments = []
    for i in range(n_coils):
        last_pt = next_pt
        frac = end_len + ((1 - 2*end_len) * (i+1)/n_coils)
        coil_dir = (-1)**i
        next_pt = (start_pt[0] + mag * frac * dir_vec[0] + coil_dir * orth_vec[0] * spring_width/2, start_pt[1] + mag * frac * dir_vec[1] + coil_dir * orth_vec[1] * spring_width/2)
        if i == n_coils-1:
            next_pt = (end_pt[0] - end_len * mag * dir_vec[0], end_pt[1] - end_len * mag * dir_vec[1])
        middle_segments.append((last_pt, next_pt))
    # Actually draw the middle segments in the correct order
    order = list(range(0, n_coils, 2)) + list(range(1, n_coils, 2))
    for i in range(len(middle_segments)):
        val = int(64 + 64 * (i // (n_coils/2)))
        draw.line(
            middle_segments[order[i]],
            (val, val, val), width=stroke_width
        )
        # Give the line some nice caps
        draw.circle(middle_segments[order[i]][0], radius=stroke_width//2, fill=(128,128,128))
        draw.circle(middle_segments[order[i]][1], radius=stroke_width//2, fill=(128,128,128))

    # Draw start end
    draw.line(
        [start_pt, middle_segments[0][0]],
        (128,128,128), width=stroke_width
    )
    # Draw end end
    draw.line(
        [middle_segments[-1][1], end_pt],
        (128,128,128), width=stroke_width
    )

def render_nice_video(sysX:MultiSystem, sysY:MultiSystem):
    # Render a video that displays all of the system values and shows cursors setting up the initial conditions
    output_path = os.path.join("output", "nice_result.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps = 60
    frame_width = 1080  # Dimensions recommended for YouTube short
    frame_height = 1920
    frame_num = 0
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_width, frame_height))

    spring_layer = Image.new("RGBA", (frame_width, frame_height))
    spring_draw = ImageDraw.Draw(spring_layer, 'RGBA')
    mass_layer = Image.new("RGBA", (frame_width, frame_height))
    mass_draw = ImageDraw.Draw(mass_layer, 'RGBA')
    label_layer = Image.new("RGBA", (frame_width, frame_height))
    label_draw = ImageDraw.Draw(label_layer, 'RGBA')
    cursor_layer = Image.new("RGBA", (frame_width, frame_height))
    cursor_draw = ImageDraw.Draw(cursor_layer, 'RGBA')

    # Load cursors
    cursor_scale = 1.5
    cursor_default = Image.open(os.path.join("graphics", "cursors", "point.png")).convert("RGBA")
    cursor_default = cursor_default.resize((int(cursor_default.width * cursor_scale), int(cursor_default.height * cursor_scale)))
    cursor_grab = Image.open(os.path.join("graphics", "cursors", "grab.png")).convert("RGBA")
    cursor_grab = cursor_grab.resize((int(cursor_grab.width * cursor_scale), int(cursor_grab.height * cursor_scale)))
    cursor_open = Image.open(os.path.join("graphics", "cursors", "open.png")).convert("RGBA")
    cursor_open = cursor_open.resize((int(cursor_open.width * cursor_scale), int(cursor_open.height * cursor_scale)))

    fnt = ImageFont.truetype(os.path.join("fonts", "16020_FUTURAM.ttf"), 64)
    smallfnt = ImageFont.truetype(os.path.join("fonts", "16020_FUTURAM.ttf"), 24)

    # Define animation schedule states
    # Mostly for the cursors animation
    # (ID, duration)
    schedule = (
        ("INTRO", 5, False),
        ("CURSORS_TO", 4, False),
        ("CURSORS_GRAB", 1, False),
        ("CURSORS_SETUP", 2, False),
        ("CURSORS_RUN", 1.1, False),
        ("CURSORS_RELEASE", 0.9, True),  # When this happens, the simulation begins at its initial conditions
        ("CURSORS_RETREAT", 2, True),
        ("CURSORS_GONE", 6, True),
        ("ZOOMING_IN", 4.3, True),
        ("ZOOMED_IN", 8.7, True),
        ("ZOOMING_OUT", 5, True),
        ("HOLD", 12, True)  # By the end of this, the video ends
    )
    state_idx = 0
    time_in_state = 0
    cursor_pos = [(-30, randint(500, 1000)) for i in range(sysX.n + sysY.n)]
    sketch_start_tidx = 0

    t_start = 0  # start for the simulation; don't change this
    t_stop = sum([s[1] for s in schedule if s[2]])  # stop time for the simulation; length of video minus any non-simulated parts
    t = np.linspace(t_start, t_stop/10, num=math.ceil(video_fps * (t_stop - t_start) + 2))
    tx, xx = sysX.get_response(t_start, t_stop, t)
    ty, xy = sysY.get_response(t_start, t_stop, t)

    # Manage audio generation
    # Each spring will generate an audio signal which will all be summed together into a final result
    # The pitch/frequency of the signal depends on the current length of the spring
    # (Based on the assumption that the spring vibrates like a string)
    # Resonant frequencies of a stretched string are described by, where fn = freq, n = num, L = length, T = tension, mu = linear density:
    # fn = (n / (2 * L)) * sqrt(T / mu)
    # The amplitude of harmonics/overtones depends on the type of excitation, so we'll ignore that for now (maybe we'll just make it triangleish later)
    # Youtube wants audio to be at 48 kHz sample rate
    # For a proper glissando effect, we will need to compensate for phase changes
    # https://stackoverflow.com/questions/3089832/sine-wave-glissando-from-one-pitch-to-another-in-numpy
    # p1 = p0 + 2 * pi * t * (f0 - f1)
    # Where p1 = phase of 1, p0 = phase of 0, t = time of match, f0 = freq of 0, f1 = freq of 1
    # For the simulation, we could probably generate everything separately from the video,
    # But since we also move the masses manually we should do it frame-by-frame
    audio_samplerate = 48000
    audio_vector = np.zeros((math.ceil(audio_samplerate * sum([s[1] for s in schedule]))))
    previous_freqs = (sysX.n + 1 + sysY.n + 1) * [0]
    current_freqs = (sysX.n + 1 + sysY.n + 1) * [0]
    previous_phases = (sysX.n + 1 + sysY.n + 1) * [0]
    current_phases = (sysX.n + 1 + sysY.n + 1) * [0]
    current_amps = (sysX.n + 1 + sysY.n + 1) * [0]
    tremolo_freqs = (sysX.n + 1 + sysY.n + 1) * [0]
    next_free_sample = 0
    generate_audio = False
    # when each frame is generated, samples up to time * audio_samplerate should be filled up
    # I'm describing it like this to try to circumvent issues with fractional numbers

    # Draw animated system
    sys_pack = [(sysX, 0, tx, xx), (sysY, np.pi / 2, ty, xy)]  # system and its axis
    tidx = 1  # tidx=0 is handled by the very end of CURSOR_RUN
    sep = 5  # default spring length or separation between masses
    low_lim = -sep * ((sysX.n - 1) / 2 + 1)
    up_lim = sep * ((sysX.n - 1) / 2 + 1)
    while True:
        # Loop for every time step
        print(f"State: {schedule[state_idx][0]}; time in state: {time_in_state:.5f}")
        # Clear the frames
        spring_layer.paste((0, 0, 0, 255), (0, 0, label_layer.width, label_layer.height))
        mass_layer.paste((0, 0, 0, 0), (0, 0, label_layer.width, label_layer.height))
        label_layer.paste((0, 0, 0, 0), (0, 0, label_layer.width, label_layer.height))
        cursor_layer.paste((0, 0, 0, 0), (0, 0, label_layer.width, label_layer.height))

        if True or schedule[state_idx][0] in ("INTRO", "CURSORS_TO"):
            # Intro; display some explanatory text
            # Top: "Given a system, you can solve for the response given initial conditions."
            # Bottom: "What if we did that, but in reverse?"
            # Focus group says that this isn't intelligiblWhat if we did that, but in reverse?e to normal people
            # Top: "Reverse-engineering a multi-degree-of-freedom mass-spring system using eigenvalue linear algebra"
            # Bottom: "For the express purpose of drawing amogus"
            cursor_draw.multiline_text((frame_width/2, 0.1*frame_height), "Reverse-engineering a\nmulti-degree-of-freedom\nmass-spring system\nusing eigenvalue linear algebra", font=fnt, fill=(255,255,255), anchor='ms', align="center")
            cursor_draw.multiline_text((frame_width/2, 0.9*frame_height), "For the express purpose\nof drawing amogus", font=fnt, fill=(255,255,255), anchor='ms', align="center")

        # Calculate camera zoom
        smoothx = 0
        if schedule[state_idx][0] == "ZOOMING_IN":
            smoothx = time_in_state / schedule[state_idx][1]
        if schedule[state_idx][0] == "ZOOMED_IN":
            smoothx = 1
        if schedule[state_idx][0] == "ZOOMING_OUT":
            smoothx = 1 - time_in_state / schedule[state_idx][1]
        # smooth = 3 * smoothx** 2 - 2 * smoothx**3
        smooth = 6 * smoothx ** 5 - 15 * smoothx ** 4 + 10 * smoothx ** 3
        zoom_mult = 1 + 7 * smooth

        def cen_trans(pt):
            # Center a point (x and y) and translate/transform it to look nice on a canvas
            # Going from a range of low_lim (negative) to up_lim (positive) to 0 to frame_width
            # Need to automatically calculate where the center of the image is on this axis
            if type(pt) != tuple:
                pt = tuple(pt)
            frame_center = (frame_width / 2, frame_height / 2)
            # Assume the length of the axis is frame_width
            orig_axis_len = up_lim - low_lim
            new_axis_len = frame_width
            len_scale = new_axis_len / orig_axis_len
            return int(zoom_mult * pt[0] * len_scale + frame_center[0]), int(frame_height - zoom_mult * pt[1] * len_scale - frame_center[1])

        # Draw springs
        for s in range(2):
            # For each of our systems
            sys, angle, t, y = sys_pack[s]
            rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

            # Draw rail
            upper_rail_points = rot_mat @ (
                    y[3, tidx] * np.array([[1, 1], [0, 0]]) + np.array([[0.025, 0.025], [-1, 1]]))
            cursor_draw.line([
                cen_trans(upper_rail_points[:, 0].flatten()),
                cen_trans(upper_rail_points[:, 1].flatten())
            ], fill=(255, 255, 255))
            lower_rail_points = rot_mat @ (
                    y[3, tidx] * np.array([[1, 1], [0, 0]]) + np.array([[-0.025, -0.025], [-1, 1]]))
            cursor_draw.line([
                cen_trans(lower_rail_points[:, 0].flatten()),
                cen_trans(lower_rail_points[:, 1].flatten())
            ], fill=(255, 255, 255))

            # Draw pin symbols
            line_hlen = 0.05 / 2
            cursor_draw.line([
                cen_trans((xx[3, tidx] - line_hlen, -line_hlen)), cen_trans((xx[3, tidx] + line_hlen, line_hlen))
            ], width=4, fill=(255, 0, 0))
            cursor_draw.line([
                cen_trans((xx[3, tidx] + line_hlen, -line_hlen)), cen_trans((xx[3, tidx] - line_hlen, line_hlen))
            ], width=4, fill=(255, 0, 0))
            cursor_draw.line([
                cen_trans((-line_hlen, xy[3, tidx] - line_hlen)), cen_trans((line_hlen, xy[3, tidx] + line_hlen))
            ], width=4, fill=(255, 0, 0))
            cursor_draw.line([
                cen_trans((line_hlen, xy[3, tidx] - line_hlen)), cen_trans((-line_hlen, xy[3, tidx] + line_hlen))
            ], width=4, fill=(255, 0, 0))

            mass_points = sys.n * [(0,0)]  # storing so we don't have to recalculate the prev_point
            for i in range(sys.n):
                # For each mass (connected to two springs but usually only needs to draw one)
                a = i + s * (sys.n + 1)  # index for audio generators
                mass_point = rot_mat @ np.array([[low_lim + sep * (i + 1) + y[i, tidx], 0]]).T

                if schedule[state_idx][0] in ("INTRO", "CURSORS_TO", "CURSORS_GRAB", "CURSORS_SETUP", "CURSORS_RUN"):
                    # Want to accelerate from this initial position to be at speed v0 when the cursor releases (position x0)
                    # x = x0 + v0 * t + 0.5 * a * t**2
                    # v = a * t
                    # Assuming constant a
                    # Given v(initial) = 0, v(final) = v0, x(final) = x0, t(final) = schedule[state_idx+1][1], solve for x(initial) and a(all)
                    run_dur = [s[1] for s in schedule if s[0] == "CURSORS_RUN"][0]
                    accel = (1.0 * sys.v0[i]/10) / run_dur  # the /10 is because the entire timescale is /10 for aesthetics. the 0.9 is just for fun
                    x_start = sys.x0[i] - 0.5 * accel * run_dur ** 2  # the end of CURSORS_RUN is tidx=0, then the simulation starts at tidx=1

                    if schedule[state_idx][0] in ("INTRO", "CURSORS_TO", "CURSORS_GRAB"):
                        mass_point = rot_mat @ np.array([[low_lim + sep * (i + 1), 0]]).T
                    elif schedule[state_idx][0] == "CURSORS_SETUP":
                        mass_point = rot_mat @ (
                                    np.array([[low_lim + sep * (i + 1), 0]]).T + np.array([[x_start * time_in_state / schedule[state_idx][1], 0]]).T)
                    elif schedule[state_idx][0] == "CURSORS_RUN":
                        mass_point = rot_mat @ (
                                    np.array([[low_lim + sep * (i + 1), 0]]).T + np.array([[x_start + 0.5 * accel * time_in_state**2, 0]]).T)

                mass_draw_point = cen_trans((mass_point[0, 0], mass_point[1, 0]))
                mass_points[i] = (mass_point[0, 0], mass_point[1, 0])
                # Draw mass
                draw_mass(mass_draw, mass_draw_point, float(zoom_mult * (5 + 5 * np.sqrt(sys.M[i, i]))))

                # Draw springs
                # We can update the spring frequencies in here too
                if i == 0:
                    # Spring connects to wall
                    spring_k = sys.K[0,0] + sys.K[i+1,i]
                    wall_point = rot_mat @ np.array([[low_lim, 0]]).T
                    spring_len = ((wall_point[0,0] - mass_point[0,0])**2 + (wall_point[1,0] - mass_point[1,0])**2)**0.5
                    spring_width = int(zoom_mult * 30 * (1 - 2*(spring_len/sep - 1) / (1 + 2*abs(spring_len/sep - 1))))
                    # int(zoom_mult * 30 * sep / (spring_len + 2))
                    draw_spring(spring_draw, cen_trans(wall_point.flatten()), mass_draw_point, spring_width, int(zoom_mult * (3 + spring_k / 1000)), n_coils=k_to_coils(spring_k))
                else:
                    # Connect to the previous mass
                    spring_k = -sys.K[i-1+1,i-1]
                    prev_point = mass_points[i-1]
                    spring_len = ((prev_point[0] - mass_point[0,0])**2 + (prev_point[1] - mass_point[1,0])**2)**0.5
                    spring_width = int(zoom_mult * 30 * (1 - 2*(spring_len/sep - 1) / (1 + 2*abs(spring_len/sep - 1))))
                    draw_spring(spring_draw, cen_trans(prev_point), mass_draw_point, spring_width, int(zoom_mult * (3 + spring_k / 1000)), n_coils=k_to_coils(spring_k))
                if generate_audio:
                    # Calculate frequency and phase of wave for the spring before this mass
                    current_freqs[a] = string_freq(spring_len, spring_k * spring_len, 0.003)
                    current_phases[a] = match_phase(previous_freqs[a], previous_phases[a], current_freqs[a],
                                                    next_free_sample / audio_samplerate)
                    #current_amps[a] = 1 / (1 + math.exp(10 * (1 - spring_len / sep)))
                    current_amps[a] = 0.1 + 0.9 * (1 - math.exp(-20 * (spring_len / sep - 1)**2))
                    tremolo_freqs[a] = (1 / 1000) * spring_k + 6
                if i == sys.n - 1:
                    # Connect to the other wall
                    next_spring_k = sys.K[-1,-1] + sys.K[-1,-2]
                    wall_point = rot_mat @ np.array([[up_lim, 0]]).T
                    spring_len = ((mass_point[0,0] - wall_point[0,0])**2 + (mass_point[1,0] - wall_point[1,0])**2)**0.5
                    spring_width = int(zoom_mult * 30 * (1 - 2*(spring_len/sep - 1) / (1 + 2*abs(spring_len/sep - 1))))
                    draw_spring(spring_draw, mass_draw_point, cen_trans(wall_point.flatten()), spring_width, int(zoom_mult * (3 + next_spring_k / 1000)), n_coils=k_to_coils(next_spring_k))
                    if generate_audio:
                        # Calculate frequency and phase of the wave for the spring after this mass
                        current_freqs[a+1] = string_freq(spring_len, next_spring_k * spring_len, 0.003)
                        current_phases[a+1] = match_phase(previous_freqs[a+1], previous_phases[a+1], current_freqs[a+1],
                                                        next_free_sample / audio_samplerate)
                        #current_amps[a+1] = 1 / (1 + math.exp(10 * (1 - spring_len / sep)))
                        current_amps[a+1] = 0.1 + 0.9 * (1 - math.exp(-20 * (spring_len / sep - 1) ** 2))
                        tremolo_freqs[a+1] = (1 / 800) * next_spring_k + 10

                # Squeezing cursor logic in here to save another loop
                if schedule[state_idx][0] in ("CURSORS_TO", "CURSORS_GRAB", "CURSORS_SETUP", "CURSORS_RUN",
                                              "CURSORS_RELEASE", "CURSORS_RETREAT"):
                    c = i + s * sys.n  # index for cursors
                    if schedule[state_idx][0] == "CURSORS_TO":
                        # Cursors move from their initial positions towards the masses
                        # If they get within a certain radius of the mass, change to the open hand version
                        dir_vec = (mass_draw_point[0] - cursor_pos[c][0], mass_draw_point[1] - cursor_pos[c][1])
                        mag = (dir_vec[0] ** 2 + dir_vec[1] ** 2) ** 0.5
                        speed = 1.5 * mag * (1 + c % 3)
                        dir_vec = (0 if mag==0 else (speed / video_fps * dir_vec[0] / mag), 0 if mag==0 else (speed / video_fps * dir_vec[1] / mag))
                        cursor_pos[c] = (cursor_pos[c][0] + dir_vec[0], cursor_pos[c][1] + dir_vec[1])
                        if mag < 50:
                            cursor_layer.paste(cursor_open, (int(cursor_pos[c][0] - cursor_open.width/2), int(cursor_pos[c][1] - cursor_open.height/2)), mask=cursor_open)
                        else:
                            cursor_layer.paste(cursor_default, (int(cursor_pos[c][0] - cursor_default.width/2), int(cursor_pos[c][1] - cursor_default.height/2)), mask=cursor_default)
                    if schedule[state_idx][0] == "CURSORS_GRAB":
                        cursor_pos[c] = mass_draw_point
                        cursor_layer.paste(cursor_grab, (int(cursor_pos[c][0] - cursor_grab.width/2), int(cursor_pos[c][1] - cursor_grab.height/2)), mask=cursor_grab)
                    if schedule[state_idx][0] in ("CURSORS_SETUP", "CURSORS_RUN"):
                        # Follow masses
                        cursor_pos[c] = mass_draw_point
                        cursor_layer.paste(cursor_grab, (int(cursor_pos[c][0] - cursor_grab.width/2), int(cursor_pos[c][1] - cursor_grab.height/2)), mask=cursor_grab)
                    if schedule[state_idx][0] == "CURSORS_RELEASE":
                        cursor_layer.paste(cursor_open, (int(cursor_pos[c][0] - cursor_open.width/2), int(cursor_pos[c][1] - cursor_open.height/2)), mask=cursor_open)
                    if schedule[state_idx][0] == "CURSORS_RETREAT":
                        cursor_goal_pos = (frame_width+100, 400 + 500*(c % 3))
                        orig_cursor_pos = cen_trans((rot_mat @ np.array([[low_lim + sep * (i + 1) + y[i, 0], 0]]).T).flatten())
                        full_goal_dist = ((cursor_goal_pos[0] - orig_cursor_pos[0])**2 + (cursor_goal_pos[1] - orig_cursor_pos[1])**2)**0.5
                        goal_vec = (cursor_goal_pos[0] - cursor_pos[c][0], cursor_goal_pos[1] - cursor_pos[c][1])
                        mag = (goal_vec[0]**2 + goal_vec[1]**2)**0.5
                        if cursor_pos[c][0] < frame_width + 10:
                            speed = 2 + 20 * abs(full_goal_dist - mag)**0.5 * (2 + c % 3)
                            dir_vec = (speed / video_fps * goal_vec[0] / mag, speed / video_fps * goal_vec[1] / mag)
                            cursor_pos[c] = (cursor_pos[c][0] + dir_vec[0], cursor_pos[c][1] + dir_vec[1])
                            cursor_layer.paste(cursor_default, (int(cursor_pos[c][0] - cursor_default.width / 2), int(cursor_pos[c][1] - cursor_default.height / 2)), mask=cursor_default)

                # Draw labels for springs and masses
                if schedule[state_idx][0] in ("INTRO", "CURSORS_TO"):
                    # Draw a label for this mass and the spring before it
                    alpha = 255
                    if schedule[state_idx][0] == "CURSORS_TO":
                        alpha = int(255 * (1 - time_in_state / schedule[state_idx][1]))
                    text_color = (255, 255, 255, alpha)
                    line_color = (255, 165, 0, alpha)
                    offset = 100  # 50 is ok
                    text_spacing_offset = 0.05 * offset
                    mass_anchor = 'mm'
                    spring_anchor = 'mm'
                    if abs(angle - 0) < 0.1:
                        # Horizontal system (horiz anchor of text)
                        mass_anchor = 'mt'
                        spring_anchor = 'ms'
                    elif abs(angle - np.pi / 2) < 0.1:
                        # Vertical system (vert anchor of text)
                        mass_anchor = 'rm'
                        spring_anchor = 'lm'
                    mass = sys.M[i,i]
                    mass_text_draw_point = (mass_draw_point[0] - offset * math.sin(angle), mass_draw_point[1] + offset * math.cos(angle))
                    label_draw.text(mass_text_draw_point,
                                     f"{mass:.2f} kg", font=smallfnt, fill=text_color, anchor=mass_anchor, align='center')
                    mass_text_draw_point = (mass_text_draw_point[0] + text_spacing_offset * math.sin(angle),
                                            mass_text_draw_point[1] - text_spacing_offset * math.cos(angle))
                    label_draw.line([mass_draw_point, mass_text_draw_point],
                                     fill=line_color, width=2)
                    if i == 0:
                        wall_point = rot_mat @ np.array([[low_lim, 0]]).T
                        spring_center = ((wall_point[0,0] + mass_points[i][0]) / 2, (wall_point[1,0] + mass_points[i][1]) / 2)
                    else:
                        spring_center = ((mass_points[i-1][0] + mass_points[i][0]) / 2, (mass_points[i-1][1] + mass_points[i][1]) / 2)
                    spring_center_draw_point = cen_trans(spring_center)
                    spring_text_draw_point = (spring_center_draw_point[0] + offset * math.sin(angle), spring_center_draw_point[1] - offset * math.cos(angle))
                    label_draw.text(spring_text_draw_point,
                                     f"{spring_k:.0f} N/m", font=smallfnt, fill=text_color, anchor=spring_anchor, align='center')
                    spring_text_draw_point = (spring_text_draw_point[0] - text_spacing_offset * math.sin(angle),
                                              spring_text_draw_point[1] + text_spacing_offset * math.cos(angle))
                    label_draw.line([spring_center_draw_point, spring_text_draw_point], fill=line_color, width=2)
                    if i == sys.n - 1:
                        # Draw a label for the spring after this mass, too
                        wall_point = rot_mat @ np.array([[up_lim, 0]]).T
                        spring_center = ((mass_points[i][0] + wall_point[0,0]) / 2, (mass_points[i][1] + wall_point[1,0]) / 2)
                        spring_center_draw_point = cen_trans(spring_center)
                        spring_text_draw_point = (spring_center_draw_point[0] + offset * math.sin(angle), spring_center_draw_point[1] - offset * math.cos(angle))
                        label_draw.text(spring_text_draw_point,
                                         f"{next_spring_k:.0f} N/m", font=smallfnt, fill=text_color, anchor=spring_anchor, align='center')
                        spring_text_draw_point = (spring_text_draw_point[0] - text_spacing_offset * math.sin(angle),
                                                  spring_text_draw_point[1] + text_spacing_offset * math.cos(angle))
                        label_draw.line([spring_center_draw_point, spring_text_draw_point], fill=line_color,
                                         width=2)

                # Draw rail block (to make it look more connected and realistic)
                block_points = rot_mat @ (y[3, tidx] * np.array([[1, 1, 1, 1], [0, 0, 0, 0]]) + np.array(
                    [  # Note that this is immediately transposed
                        [-0.05, -0.15],
                        [-0.05, 0.15],
                        [0.05, 0.15],
                        [0.05, -0.15]
                    ]).T)
                mass_draw.polygon([cen_trans(block_points[:, p].flatten()) for p in range(4)],
                                  fill=(112, 128, 144), outline=(119, 136, 153+20), width=4)

        # We're all done with the individual masses/springs!

        # Draw sketch trajectory
        if schedule[state_idx][0] in ("ZOOMED_IN", "ZOOMING_OUT", "HOLD"):
            for tt in range(sketch_start_tidx,tidx):
                cursor_draw.line([cen_trans((xx[3,tt-1], xy[3,tt-1])), cen_trans((xx[3,tt], xy[3,tt]))], (0,255,0), width=4)

        # Draw "pencil"
        mass_draw.circle(cen_trans((xx[3,tidx], xy[3,tidx])),  radius=4, fill=(0,0,255))

        # Collapse layers
        spring_layer.paste(mass_layer, mask=mass_layer)
        spring_layer.alpha_composite(label_layer, (0,0), (0,0))
        spring_layer.paste(cursor_layer, mask=cursor_layer)
        out_arr = np.array(spring_layer)
        out.write(np.stack((out_arr[:,:,2], out_arr[:,:,1], out_arr[:,:,0]), axis=2))

        # Generate the audio associated with this frame
        # print(f"We're on frame {frame_num}, so we need to end with the next_free_sample (currently {next_free_sample}) at {(frame_num + 1) * (1/video_fps) * audio_samplerate}")
        # print(current_freqs)
        if generate_audio:
            while next_free_sample < (frame_num + 1) * (1/video_fps) * audio_samplerate:
                # Fill audio samples with the current frequencies
                vol_mod = 1.0
                if schedule[state_idx][0] not in ("INTRO", "CURSORS_TO", "CURSORS_GRAB"):
                    if schedule[state_idx][0] == "CURSORS_SETUP":
                        vol_mod = time_in_state / schedule[state_idx][1]
                    for i in range(len(current_freqs)):
                        if current_freqs[i] != 0:
                            # audio_vector[next_free_sample] += np.cos(2 * np.pi * current_freqs[i] * next_free_sample / audio_samplerate + current_phases[i])
                            tremolo = 0.7 + 0.3 * math.sin(2 * math.pi * tremolo_freqs[i] * next_free_sample / audio_samplerate)
                            vibrato = 0.5 * math.sin(2 * math.pi * tremolo_freqs[i] * next_free_sample / audio_samplerate)
                            audio_vector[next_free_sample] += vol_mod * current_amps[i] * tremolo * guitar_string_wave(30, current_freqs[i], current_phases[i] + vibrato, next_free_sample / audio_samplerate)
                next_free_sample += 1
            previous_freqs = [current_freqs[i] for i in range(len(current_freqs))]  # need deep copies of these lists
            previous_phases = [current_phases[i] for i in range(len(current_phases))]

        # Increase timesteps
        frame_num += 1
        time_in_state += 1 / video_fps

        # Handle state transition
        if abs(time_in_state - schedule[state_idx][1]) < (0.5 * 1 / video_fps):  # some little fudge factor
            # We've spent enough time in this state; move to the next or end it all
            if schedule[state_idx][0] == schedule[-1][0]:
                break
            time_in_state = 0
            state_idx += 1
            if schedule[state_idx][0] == "ZOOMED_IN":
                sketch_start_tidx = tidx
        else:
            # Update tidx if we're in the right states
            if schedule[state_idx][0] in ("CURSORS_RELEASE", "CURSORS_RETREAT", "CURSORS_GONE", "ZOOMING_IN", "ZOOMED_IN", "ZOOMING_OUT", "HOLD"):
                print(f"Rendering simulation frame {tidx} (t={t[tidx]:.05f}); {100 * tidx / len(t):.01f}% through simulation")
                tidx += 1

    # All done! Release our resources
    print("Releasing video resources")
    out.release()
    cv2.destroyAllWindows()

    # And save the audio
    if generate_audio:
        print("Saving audio!")
        wavio.write(os.path.join("output", "audio.wav"), audio_vector, audio_samplerate, sampwidth=1, scale=np.max(np.abs(audio_vector)))


if __name__ == "__main__":
    sysX = MultiSystem(
        n = 7,
        M = np.diag([3.0978, 4.7058, 0.2062, 0.3439, 18.8495, 3.2583, 4.9459]),
        B = np.zeros([7,7]),
        K = 10**3 * np.array([
            [3.6019, -1.9407, 0, 0, 0, 0, 0],
            [-1.9407, 1.9760, -0.0353, 0, 0, 0, 0],
            [0, -0.0353, 0.1149, -0.0796, 0, 0, 0],
            [0, 0, -0.0796, 0.2657, -0.1861, 0, 0],
            [0, 0, 0, -0.1861, 2.2605, -2.0743, 0],
            [0, 0, 0, 0, -2.0743, 4.6377, -2.5634],
            [0, 0, 0, 0, 0, -2.5634, 5.2983]
        ]),
        x0 = np.array([0.2263, -0.1049, 0.2320, 0.3510, 0.4655, -1.3546, 0.0236]).T,
        v0 = np.array([-13.7577, 12.4859, 12.6717, -0.7816, -6.5906, 40.1957, 7.8644]).T
    )

    sysY = MultiSystem(
        n = 7,
        M = np.zeros([7,7]),
        B = np.zeros([7,7]),
        K = np.zeros([7,7]),
        x0 = np.array([-0.6677, 0.4141, 0.9281, -0.0302, -0.5509, 2.2719, 1.6460]).T,
        v0 = np.array([-37.4518, 8.4820, -6.0754, -6.1489, -5.1286, 22.6085, 3.6750]).T
    )
    sysY.M = sysX.M
    sysY.K = sysX.K

    # pyplot_animate_2_systems(sysX, sysY, 0, 4)

    render_nice_video(sysX, sysY)

    # draw_spring(None, (0,0), (100,0), 5, 1, n_coils=5)


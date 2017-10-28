# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, aspect='equal')

# ax2.add_patch(patches.Rectangle((0, 0), 512, 660, fill=False)

# plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')

# ax1.add_patch(
#     patches.Rectangle(
#         (0.1, 0.1),   # (x,y)
#         0.5,          # width
#         0.5,          # height
#     )
# )

ax1.add_patch(patches.Rectangle((0, 0), 0.512, 0.660, fill = False))
# zone 1 through 4 arms
ax1.add_patch(patches.Rectangle((0, 0.160), 0.200, 0.070, fill = False))
ax1.add_patch(patches.Rectangle((0, 0), 0.200, 0.160, fill = False))
ax1.add_patch(patches.Rectangle((0.330, 0.160), 0.182, 0.080, fill = False))
ax1.add_patch(patches.Rectangle((0.330, 0.0), 0.182, 0.160, fill = False))
# zone 5 and 17
ax1.add_patch(patches.Rectangle((0, 0.220), 0.512, 0.08, fill = False))
# zone 6 through 10 main body
ax1.add_patch(patches.Rectangle((0,     0.300), 0.256, 0.060, fill = False))
ax1.add_patch(patches.Rectangle((0.256, 0.300), 0.256, 0.060, fill = False))
ax1.add_patch(patches.Rectangle((0,     0.370), 0.225, 0.080, fill = False))
ax1.add_patch(patches.Rectangle((0.225, 0.370), 0.050, 0.080, fill = False))
ax1.add_patch(patches.Rectangle((0.275, 0.370), 0.237, 0.080, fill = False))
# zone 11 through 16 legs
ax1.add_patch(patches.Rectangle((0, 0.450), 0.256, 0.075, fill = False))
ax1.add_patch(patches.Rectangle((0.256, 0.450), 0.256, 0.075, fill = False))
ax1.add_patch(patches.Rectangle((0, 0.525), 0.256, 0.075, fill = False))
ax1.add_patch(patches.Rectangle((0.256, 0.525), 0.256, 0.075, fill = False))
ax1.add_patch(patches.Rectangle((0, 0.600), 0.256, 0.060, fill = False))
ax1.add_patch(patches.Rectangle((0.256, 0.600), 0.256, 0.060, fill = False))

plt.show()



import cadquery as cq
import math

ARM_R = 55.0
BASE_T = 3.0
STANDOFF_CLEAR = 5.0
PCB_L, PCB_W, PCB_T = 66.0, 52.0, 1.6
PCB_GAP = 0.5
RAIL_T = 2.4
RAIL_ABOVE_PCB = 1.2
END_MARGIN = 14.0
WING = 13.0
BOSS_OD = 7.0
PILOT_D = 2.5
STRAP_W, STRAP_SLOT_T = 26.0, 3.5
TIE_W, TIE_T = 4.5, 2.2
END_WALL_T = 3.0
WALL_HALF_W = 28.9
FRONT_HOLE_D = 20.0
FRONT_HOLE_ZC = BASE_T + 0.5 + FRONT_HOLE_D / 2
FRONT_WALL_TOP = FRONT_HOLE_ZC + FRONT_HOLE_D / 2 + 3.0
REAR_SLOT_W, REAR_SLOT_H = 20.0, 5.0
REAR_SLOT_ZC = 18.0
REAR_WALL_TOP = REAR_SLOT_ZC + REAR_SLOT_H / 2 + 3.0

L = PCB_L + 2 * END_MARGIN
INNER_HALF_W = PCB_W / 2 + PCB_GAP
W = 2 * (INNER_HALF_W + RAIL_T + WING)

z_pcb_bot = BASE_T + STANDOFF_CLEAR
z_pcb_top = z_pcb_bot + PCB_T
z_rail_top = z_pcb_top + RAIL_ABOVE_PCB

holes_kicad = [(3.5, 3.5), (3.5, 48.5), (44.6, 3.5), (44.6, 48.5)]
holes = [(bx - PCB_L / 2, PCB_W / 2 - by) for bx, by in holes_kicad]


def arm_cyl(r, length=L + 40):
    return (cq.Workplane("YZ").workplane(offset=-length / 2)
            .center(0, -ARM_R).circle(r).extrude(length))


envelope = cq.Workplane("XY").box(L, W, 80, centered=(True, True, False)).translate((0, 0, -60))
shell = arm_cyl(ARM_R + BASE_T).cut(arm_cyl(ARM_R)).intersect(envelope)

body = shell
for hx, hy in holes:
    boss = cq.Workplane("XY").center(hx, hy).circle(BOSS_OD / 2).extrude(z_pcb_bot + 30).translate((0, 0, -30))
    body = body.union(boss)

rail_len = PCB_L + 2 * PCB_GAP
for s in (1, -1):
    yc = s * (INNER_HALF_W + RAIL_T / 2)
    rail = (cq.Workplane("XY").box(rail_len, RAIL_T, z_rail_top + 30, centered=(True, True, False))
            .translate((0, yc, -30)))
    rail = rail.edges("|X and >Z").chamfer(0.8)
    body = body.union(rail)

for s in (1, -1):
    xc = s * (L / 2 - 6.5)
    pad = (cq.Workplane("XY").box(8, 16, BASE_T + 5 + 30, centered=(True, True, False))
           .translate((xc, 0, -30)))
    pad = pad.edges("|Y and >Z").fillet(2.5)
    body = body.union(pad)
    tie = cq.Workplane("XZ").center(xc, BASE_T + 2.2).rect(TIE_W, TIE_T).extrude(30, both=True)
    body = body.cut(tie)

for sx, top in ((1, FRONT_WALL_TOP),):
    xc = sx * (L / 2 - END_WALL_T / 2)
    wall = (cq.Workplane("YZ").workplane(offset=xc - END_WALL_T / 2)
            .center(0, (top - 30) / 2).rect(2 * WALL_HALF_W, top + 30).extrude(END_WALL_T))
    wall = wall.edges("|X and >Z").fillet(6.0)
    body = body.union(wall)

body = body.cut(arm_cyl(ARM_R))
body = body.intersect(cq.Workplane("XY").box(L, W, 200, centered=(True, True, True)))

front_x = L / 2 - END_WALL_T / 2
front_hole = (cq.Workplane("YZ").workplane(offset=front_x - END_WALL_T / 2 - 0.2)
              .center(0, FRONT_HOLE_ZC).circle(FRONT_HOLE_D / 2).extrude(END_WALL_T + 5))
body = body.cut(front_hole)

for hx, hy in holes:
    pilot = cq.Workplane("XY").center(hx, hy).circle(PILOT_D / 2).extrude(80).translate((0, 0, -40))
    body = body.cut(pilot)

for sx in (1, -1):
    for sy in (1, -1):
        xc = sx * (L / 2 - 4 - STRAP_W / 2)
        yc = sy * (INNER_HALF_W + RAIL_T + WING / 2 + 0.5)
        slot = (cq.Workplane("XY").center(xc, yc).slot2D(STRAP_W, STRAP_SLOT_T, angle=0)
                .extrude(120).translate((0, 0, -60)))
        body = body.cut(slot)

for vy in (-9.0, 9.0):
    vent = (cq.Workplane("XY").center(-4.0, vy).slot2D(44, 6, angle=0)
            .extrude(40).translate((0, 0, -20)))
    body = body.cut(vent)

try:
    body = body.edges("|X").edges(cq.selectors.BoxSelector((-L, -W, -40), (L, W, BASE_T + 0.5))).fillet(1.0)
except Exception as e:
    print("fillet skipped:", e)

cq.exporters.export(body, "forearm_pcb_mount.stl", tolerance=0.05, angularTolerance=0.1)
cq.exporters.export(body, "forearm_pcb_mount.step")
bb = body.val().BoundingBox()
print(f"bbox: {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm, volume {body.val().Volume()/1000:.1f} cm^3")
print("front hole zc:", FRONT_HOLE_ZC, "front wall top:", FRONT_WALL_TOP, "rear slot zc:", REAR_SLOT_ZC, "rear wall top:", REAR_WALL_TOP)
print("pcb bottom z:", z_pcb_bot, "pcb top z:", z_pcb_top, "rail top z:", z_rail_top)
print("holes (model xy):", holes)

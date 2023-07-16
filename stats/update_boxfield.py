"""
The script is intended to update boxfield.py.

boxfield.py is used in surface_function to
create surface plots of fenics functions defined
on a regular grid of the unit square.
"""

file = open("boxfield.py", "r")
Lines = file.readlines()
new_lines = []

linebreak = False
for line in Lines:

	new_line = line

	if linebreak == True:
		new_line = line[:-1] + ")\n"
		linebreak = False

	if line.find("print ") > -1:
		new_line = line.replace("print ", "print(")
		new_line = new_line[:-1] + ")\n"

	if line.find("print") > -1 and line.find("\\") > -1:
		linebreak = True
		new_line = line.replace("print", "print(")


	# manual fixes
	if new_line.find("str(g3)") > -1:
		new_line = "    print( 'g3=%s' % str(g3))\n"

	if new_line.find("interpolate(u, FunctionSpace(fenics_mesh, 'P', 1))"):
		new_line = new_line.replace("interpolate(u, FunctionSpace(fenics_mesh, 'P', 1))", \
					"fenics.interpolate(fenics_function, fenics.FunctionSpace(fenics_mesh, 'P', 1))")

	new_lines.append(new_line)


file = open("boxfield.py", "w")
file.writelines(new_lines)
file.close()

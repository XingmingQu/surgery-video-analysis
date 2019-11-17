import turtle, math

# draw a line from (x1, y1) to (x2, y2)
def drawLine (ttl, x1, y1, x2, y2):
	ttl.penup()
	ttl.goto (x1, y1)
	ttl.pendown()
	ttl.goto (x2, y2)
	ttl.penup()

def main():
	# put label on top of page
	turtle.title ('Geometric Figures')

	# setup screen size
	turtle.setup (800, 800, 0, 0)

	# create a turtle object
	ttl = turtle.Turtle()

	# draw a line
	ttl.color ('gold4')
	drawLine (ttl, -200, -10, 325, -10)
	drawLine (ttl, -200, -15, 325, -15)

	# persist drawing
	turtle.done()


main()
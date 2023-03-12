a = ""
for x in range(1,20):
	if x<10:
		a += str(x)+".7z.00"+str(x) + "+"
	else:
		a += str(x)+".7z.0"+str(x) + "+"

print(a)
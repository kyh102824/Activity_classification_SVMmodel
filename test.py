count = 0
ex = []
for i in range(3) :
    data = []
    for j in range(3) :
        data.append(j + count)
    ex.append(data)
    count = count + 1

print(ex)
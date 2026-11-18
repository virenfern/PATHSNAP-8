from itertools import cycle

c1 = bytes.fromhex("ef92fb88c2bd1260a5bfbcf0798bb52acc413dfe82cc38aee1f1da18876cfb7979a5b969dbd0bbf4045d74a6661908359fe12363c1b49c7e8904a80b88")
c2 = bytes.fromhex("ef92fb88c2bd1267aab5c1f0108fb527c9413287fee91b80c3b99b39a109993b7ae4f860bee8dccc237c55a81d3a1b04bed65c27d2a0d823931ecd65ec")

crib = b"RETRO::ACK::STATION="
key = bytes(a ^ b for a, b in zip(c1, crib))
guess = b"your guessed text"



# A crib reveals only the keystream bytes at those positions.
print("crib-derived key bytes:", key.hex())
print("message 1 prefix:", bytes(a ^ b for a, b in zip(c1, key)))

# Same keystream means plaintext1 XOR plaintext2 = ciphertext1 XOR ciphertext2.
print("P1 XOR P2:", bytes(a ^ b for a, b in zip(c1, c2)).hex())
offset = 20
guess = b"your guessed text"

other = bytes(
    c1[offset + i] ^ c2[offset + i] ^ guess[i]
    for i in range(min(len(guess), len(c1) - offset))
)
print(other)

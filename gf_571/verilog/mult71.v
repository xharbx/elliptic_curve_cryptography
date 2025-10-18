`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2) Multiplier: 71-bit × 71-bit
// Polynomial multiplication, no reduction
// One-stage pipeline (registered output)
//////////////////////////////////////////////////////////////////////////////////
module mult71 (
    input  wire        clk,
    input  wire [70:0] a,
    input  wire [70:0] b,
    output reg  [141:0] d
);

    // Partial products
    wire [141:0] pp [70:0];

    genvar i;
    generate
        for (i = 0; i < 71; i = i + 1) begin : gen_pp
            assign pp[i] = b[i] ? (a << i) : 142'b0;
        end
    endgenerate

    wire [141:0] d_comb;

    // XOR all partial products
    assign d_comb = pp[0]  ^ pp[1]  ^ pp[2]  ^ pp[3]  ^ pp[4]  ^ pp[5]  ^
                    pp[6]  ^ pp[7]  ^ pp[8]  ^ pp[9]  ^ pp[10] ^ pp[11] ^
                    pp[12] ^ pp[13] ^ pp[14] ^ pp[15] ^ pp[16] ^ pp[17] ^
                    pp[18] ^ pp[19] ^ pp[20] ^ pp[21] ^ pp[22] ^ pp[23] ^
                    pp[24] ^ pp[25] ^ pp[26] ^ pp[27] ^ pp[28] ^ pp[29] ^
                    pp[30] ^ pp[31] ^ pp[32] ^ pp[33] ^ pp[34] ^ pp[35] ^
                    pp[36] ^ pp[37] ^ pp[38] ^ pp[39] ^ pp[40] ^ pp[41] ^
                    pp[42] ^ pp[43] ^ pp[44] ^ pp[45] ^ pp[46] ^ pp[47] ^
                    pp[48] ^ pp[49] ^ pp[50] ^ pp[51] ^ pp[52] ^ pp[53] ^
                    pp[54] ^ pp[55] ^ pp[56] ^ pp[57] ^ pp[58] ^ pp[59] ^
                    pp[60] ^ pp[61] ^ pp[62] ^ pp[63] ^ pp[64] ^ pp[65] ^
                    pp[66] ^ pp[67] ^ pp[68] ^ pp[69] ^ pp[70];

    // One pipeline stage
    always @(posedge clk) begin
        d <= d_comb;
    end

endmodule

#ifndef __TCP_BENCH_CLIENT_HPP__
#define __TCP_BENCH_CLIENT_HPP__

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sys/types.h>
#include <netinet/in.h>
#include <string>
#include <arpa/inet.h>
#include <iostream>

#define TTCPORT 40057

struct TCPTimelineClient {
    int sock;

    TCPTimelineClient() {
        sock = -1;
    }

    void notify(int e) {
        if (sock == -1) {
            sock = socket(AF_INET,SOCK_DGRAM,0);
            if(sock < 0)
                perror("cannot open socket");
        }
        //printf("sending ttc to %s\n", getenv("TTC_ADDR"));

        char const *ttc_str = getenv("TTC_ADDR");
        if (!ttc_str) return;

        sockaddr_in servaddr;
        memset(&servaddr, 0, sizeof(servaddr));
        servaddr.sin_family = AF_INET;
        servaddr.sin_addr.s_addr = inet_addr(ttc_str);
        servaddr.sin_port = htons(TTCPORT);

        std::string s = std::to_string(e);

        if (sendto(sock, s.c_str(), s.size()+1, 0,
               (sockaddr*)&servaddr, sizeof(servaddr)) < 0){
            perror("cannot send message");
            return;
        }
    }


    /*
    int sock;
    public:
    void connect_to(std::string address) {
        struct sockaddr_in serv_addr;

        if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            perror("Socket creation error");
        }

        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(PORT);

        // Convert IPv4 and IPv6 addresses from text to binary form
        std::string add = address.empty() ? "127.0.0.1" : address.c_str();
        if(inet_pton(AF_INET, add.c_str(), &serv_addr.sin_addr) <= 0) {
            std::cerr << "TCPTimelineClient: Invalid address/ Address not supported"
                << add << std::endl;
            return;
        }
    
        if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            std::cerr << "TCPTimelineClient: Connection to " << add << " failed. "
                << "Continuing without it." << std::endl;
            sock = -1;
        }
    }

    void notify(int e) {
        if (sock < 1) {
            //std::cerr << "TCPTimelineClient: Socket connection failed. Event not sent."
            //    << std::endl;
            return;
        }

        send(sock, &e, sizeof(int), 0);
    }
    */
};


TCPTimelineClient ttc;

#endif
